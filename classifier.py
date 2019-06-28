from sklearn.metrics import classification_report, accuracy_score
from collections import OrderedDict
from privacy.analysis.rdp_accountant import compute_rdp
from privacy.analysis.rdp_accountant import get_privacy_spent
from privacy.optimizers import dp_optimizer
import theano.tensor as T
import tensorflow as tf
import numpy as np
import lasagne
import theano
import argparse

# Compatibility with tf 1 and 2 APIs
try:
  AdamOptimizer = tf.train.AdamOptimizer
except:  # pylint: disable=bare-except
  AdamOptimizer = tf.optimizers.Adam  # pylint: disable=invalid-name

# optimal sigma values for RDP mechanism for the default batch size, training set size, delta and sampling ratio.
noise_multiplier = {0.01:525, 0.05:150, 0.1:70, 0.5:13.8, 1:7, 5:1.669, 10:1.056, 50:0.551, 100:0.445, 500:0.275, 1000:0.219}


class EpsilonPrintingTrainingHook(tf.estimator.SessionRunHook):
  """Training hook to print current value of epsilon after an epoch."""

  def __init__(self, ledger):
    """Initalizes the EpsilonPrintingTrainingHook.
    Args:
      ledger: The privacy ledger.
    """
    self._samples, self._queries = ledger.get_unformatted_ledger()

  def end(self, session):
    orders = [1 + x / 10.0 for x in range(1, 100)] + list(range(12, 64))
    samples = session.run(self._samples)
    queries = session.run(self._queries)
    formatted_ledger = privacy_ledger.format_ledger(samples, queries)
    rdp = compute_rdp_from_ledger(formatted_ledger, orders)
    eps = get_privacy_spent(orders, rdp, target_delta=1e-5)[0]
    print('For delta=1e-5, the current epsilon is: %.2f' % eps)


def iterate_minibatches(inputs, targets, batch_size, shuffle=True):
    assert len(inputs) == len(targets)
    if shuffle:
        indices = np.arange(len(inputs))
        np.random.shuffle(indices)

    start_idx = None
    for start_idx in range(0, len(inputs) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield inputs[excerpt], targets[excerpt]

    if start_idx is not None and start_idx + batch_size < len(inputs):
        excerpt = indices[start_idx + batch_size:] if shuffle else slice(start_idx + batch_size, len(inputs))
        yield inputs[excerpt], targets[excerpt]


def get_nn_model(n_in, n_hidden, n_out, non_linearity):
    net = dict()
    non_lin = lasagne.nonlinearities.tanh
    if non_linearity == 'relu':
        non_lin = lasagne.nonlinearities.rectify
    net['input'] = lasagne.layers.InputLayer((None, n_in))
    net['fc'] = lasagne.layers.DenseLayer(
        net['input'],
        num_units=n_hidden,
        nonlinearity=non_lin)
    net['fc2'] = lasagne.layers.DenseLayer(
        net['fc'],
        num_units=n_hidden,
        nonlinearity=non_lin)
    net['output'] = lasagne.layers.DenseLayer(
        net['fc2'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net


def get_softmax_model(n_in, n_out):
    net = dict()
    net['input'] = lasagne.layers.InputLayer((None, n_in))
    net['output'] = lasagne.layers.DenseLayer(
        net['input'],
        num_units=n_out,
        nonlinearity=lasagne.nonlinearities.softmax)
    return net


def train(dataset, n_hidden=50, batch_size=100, epochs=100, learning_rate=0.01, model='nn', l2_ratio=1e-7,
        silent=True, non_linearity='relu', privacy='no_privacy', dp = 'dp', epsilon=0.5, delta=1e-5):
    train_x, train_y, test_x, test_y = dataset
    n_in = train_x.shape[1]
    n_out = len(np.unique(train_y))

    if batch_size > len(train_y):
        batch_size = len(train_y)

    #print('Building model with {} training data, {} classes...'.format(len(train_x), n_out))
    input_var = T.matrix('x')
    target_var = T.ivector('y')
    if model == 'nn':
        #print('Using neural network...')
        net = get_nn_model(n_in, n_hidden, n_out, non_linearity)
    else:
        #print('Using softmax regression...')
        net = get_softmax_model(n_in, n_out)

    net['input'].input_var = input_var
    output_layer = net['output']

    # create loss function
    prediction = lasagne.layers.get_output(output_layer)
    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
    loss = loss.mean() + l2_ratio * lasagne.regularization.regularize_network_params(output_layer, lasagne.regularization.l2)
    # create parameter update expressions
    params = lasagne.layers.get_all_params(output_layer, trainable=True)

    updates = lasagne.updates.adam(loss, params, learning_rate=learning_rate)

    train_fn = theano.function([input_var, target_var], loss, updates=updates, allow_input_downcast=True)
    test_prediction = lasagne.layers.get_output(output_layer, deterministic=True)
    test_fn = theano.function([input_var], test_prediction, allow_input_downcast=True)

    #print('Training...')
    train_loss = 0
    for epoch in range(epochs):
        loss_ = 0
        for input_batch, target_batch in iterate_minibatches(train_x, train_y, batch_size):
            loss_ += train_fn(input_batch, target_batch)
        train_loss = loss_
        loss_ = round(loss_, 3)
        if not silent:
            print('Epoch {}, train loss {}'.format(epoch, loss_))

    
    pred_y = []
    for input_batch, _ in iterate_minibatches(train_x, train_y, batch_size, shuffle=False):
        pred = test_fn(input_batch)
        pred_y.append(np.argmax(pred, axis=1))
    pred_y = np.concatenate(pred_y)

    train_acc = accuracy_score(train_y, pred_y)

    if not silent:
        print('Training Accuracy: {}'.format(accuracy_score(train_y, pred_y)))
        #print(classification_report(train_y, pred_y))

    if test_x is not None:
        #print('Testing...')
        pred_y = []
        pred_scores = []

        if batch_size > len(test_y):
            batch_size = len(test_y)

        for input_batch, _ in iterate_minibatches(test_x, test_y, batch_size, shuffle=False):
            pred = test_fn(input_batch)
            pred_y.append(np.argmax(pred, axis=1))
            pred_scores.append(pred)
        pred_y = np.concatenate(pred_y)
        pred_scores = np.concatenate(pred_scores)
        test_acc = accuracy_score(test_y, pred_y)
        if not silent:
            print('Testing Accuracy: {}'.format(accuracy_score(test_y, pred_y)))
            #print(classification_report(test_y, pred_y))

        return output_layer, pred_y, pred_scores, train_loss, train_acc, test_acc


def get_predictions(predictions):
    pred_y, pred_scores = [], []
    val = next(predictions, None)
    while val is not None:
        pred_y.append(val['classes'])
        pred_scores.append(val['probabilities'])
        val = next(predictions, None)
    return np.array(pred_y), np.matrix(pred_scores)


def get_model(features, labels, mode, params):
    n, n_in, n_hidden, n_out, non_linearity, model, privacy, dp, epsilon, delta, batch_size, learning_rate, l2_ratio, epochs = params
    if model == 'nn':
        #print('Using neural network...')
        input_layer = tf.reshape(features['x'], [-1, n_in])
        y = tf.keras.layers.Dense(n_hidden, activation=non_linearity, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio)).apply(input_layer)
        y = tf.keras.layers.Dense(n_hidden, activation=non_linearity, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio)).apply(y)
        logits = tf.keras.layers.Dense(n_out, activation=tf.nn.softmax, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio)).apply(y)
    else:
        #print('Using softmax regression...')
        input_layer = tf.reshape(features['x'], [-1, n_in])
        logits = tf.keras.layers.Dense(n_out, activation=tf.nn.softmax, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio)).apply(input_layer)
    
    predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      #"probabilities": tf.nn.softmax(logits, name="softmax_tensor")
      "probabilities": logits
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    vector_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
    scalar_loss = tf.reduce_mean(vector_loss)

    if mode == tf.estimator.ModeKeys.TRAIN:

        if privacy == 'grad_pert':
            C = 1 # Clipping Threshold
            sigma = 0.
            if dp == 'adv_cmp':
                sigma = np.sqrt(epochs * np.log(2.5 * epochs / delta)) * (np.sqrt(np.log(2 / delta) + 2 * epsilon) + np.sqrt(np.log(2 / delta))) / epsilon # Adv Comp
            elif dp == 'zcdp':
                sigma = np.sqrt(epochs / 2) * (np.sqrt(np.log(1 / delta) + epsilon) + np.sqrt(np.log(1 / delta))) / epsilon # zCDP
            elif dp == 'rdp':
                sigma = noise_multiplier[epsilon]
            elif dp == 'dp':
                sigma = epochs * np.sqrt(2 * np.log(1.25 * epochs / delta)) / epsilon # DP
            print(sigma)
    
            optimizer = dp_optimizer.DPAdamGaussianOptimizer(
                            l2_norm_clip=C,
                            noise_multiplier=sigma,
                            num_microbatches=batch_size,
                            learning_rate=learning_rate,
                            ledger=None)
            opt_loss = vector_loss
        else:
            optimizer = AdamOptimizer(learning_rate=learning_rate)
            opt_loss = scalar_loss
        global_step = tf.train.get_global_step()
        train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=scalar_loss,
                                          train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy':
                tf.metrics.accuracy(
                    labels=labels,
                     predictions=predictions["classes"])
        }

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=scalar_loss,
                                          eval_metric_ops=eval_metric_ops)


def train_private(dataset, n_hidden=50, batch_size=100, epochs=100, learning_rate=0.01, model='nn', l2_ratio=1e-7,
        silent=True, non_linearity='relu', privacy='no_privacy', dp = 'dp', epsilon=0.5, delta=1e-5):
    train_x, train_y, test_x, test_y = dataset

    n_in = train_x.shape[1]
    n_out = len(np.unique(train_y))

    if batch_size > len(train_y):
        batch_size = len(train_y)

    classifier = tf.estimator.Estimator(
    		model_fn=get_model, 
    		params = [
    			train_x.shape[0], 
    			n_in, 
    			n_hidden, 
    			n_out, 
    			non_linearity, 
    			model, 
    			privacy, 
    			dp, 
    			epsilon, 
    			delta, 
    			batch_size, 
    			learning_rate, 
    			l2_ratio, 
    			epochs
    		])

    train_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_x},
        y=train_y,
        batch_size=batch_size,
        num_epochs=epochs,
        shuffle=True)
    test_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_x},
        y=test_y,
        num_epochs=1,
        shuffle=False)
    train_eval_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_x},
        y=train_y,
        num_epochs=1,
        shuffle=False)
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': test_x},
        num_epochs=1,
        shuffle=False)

    steps_per_epoch = train_x.shape[0] // batch_size
    orders = [1 + x / 100.0 for x in range(1, 1000)] + list(range(12, 1200))
    rdp = compute_rdp(batch_size / train_x.shape[0], noise_multiplier[epsilon], epochs * steps_per_epoch, orders)
    eps, _, opt_order = get_privacy_spent(orders, rdp, target_delta=delta)
    print('\nFor delta= %.5f' % delta, ',the epsilon is: %.2f\n' % eps)
    
    for epoch in range(1, epochs + 1):
        classifier.train(input_fn=train_input_fn, steps=steps_per_epoch)
    
        if not silent:
            eval_results = classifier.evaluate(input_fn=train_eval_input_fn)
            print('Train loss after %d epochs is: %.3f' % (epoch, eval_results['loss']))

    eval_results = classifier.evaluate(input_fn=train_eval_input_fn)
    train_acc = eval_results['accuracy']
    train_loss = eval_results['loss']
    if not silent:
        print('Train accuracy is: %.3f' % (train_acc))

    eval_results = classifier.evaluate(input_fn=test_eval_input_fn)
    test_acc = eval_results['accuracy']
    if not silent:
        print('Test accuracy is: %.3f' % (test_acc))

    predictions = classifier.predict(input_fn=pred_input_fn)
    
    pred_y, pred_scores = get_predictions(predictions)

    return classifier, pred_y, pred_scores, train_loss, train_acc, test_acc


def load_dataset(train_feat, train_label, test_feat=None, test_label=None):
    train_x = np.genfromtxt(train_feat, delimiter=',', dtype='float32')
    train_y = np.genfromtxt(train_label, dtype='int32')
    min_y = np.min(train_y)
    train_y -= min_y
    if test_feat is not None and test_label is not None:
        test_x = np.genfromtxt(train_feat, delimiter=',', dtype='float32')
        test_y = np.genfromtxt(train_label, dtype='int32')
        test_y -= min_y
    else:
        test_x = None
        test_y = None
    return train_x, train_y, test_x, test_y


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('train_feat', type=str)
    parser.add_argument('train_label', type=str)
    parser.add_argument('--test_feat', type=str, default=None)
    parser.add_argument('--test_label', type=str, default=None)
    parser.add_argument('--model', type=str, default='nn')
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=100)
    parser.add_argument('--n_hidden', type=int, default=50)
    parser.add_argument('--epochs', type=int, default=100)
    args = parser.parse_args()
    print(vars(args))
    dataset = load_dataset(args.train_feat, args.train_label, args.test_feat, args.train_label)
    train(dataset,
          model=args.model,
          learning_rate=args.learning_rate,
          batch_size=args.batch_size,
          n_hidden=args.n_hidden,
          epochs=args.epochs)


if __name__ == '__main__':
    main()
