from sklearn.metrics import classification_report, accuracy_score
from collections import OrderedDict
from privacy.optimizers import dp_optimizer
import tensorflow as tf
import numpy as np
import argparse

# Compatibility with tf 1 and 2 APIs
try:
  AdamOptimizer = tf.train.AdamOptimizer
except:  # pylint: disable=bare-except
  AdamOptimizer = tf.optimizers.Adam  # pylint: disable=invalid-name


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
            alpha = 1 # Parameter for Renyi Divergence
            C = 1 # Clipping Threshold
            _q = batch_size / n # Sampling Ratio
            _T = epochs * n / batch_size # Number of Steps
            sigma = 0.
            if dp == 'adv_cmp':
                sigma = _q * np.sqrt(2 * _T * np.log(1.25 / delta) * np.log(_T / delta)) / epsilon # Adv Comp
            elif dp == 'zcdp':
                sigma = _q * np.sqrt(2 * _T * np.log(1.25 / delta)) / epsilon # zCDP
            elif dp == 'rdp':
                sigma = _q * np.sqrt(alpha * _T / 2 / epsilon) # RDP
            elif dp == 'dp':
                sigma = _q * _T * np.sqrt(2 * np.log(1.25 / delta)) / epsilon # DP
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


def train(dataset, hold_out_train_data=None, n_hidden=50, batch_size=100, epochs=100, learning_rate=0.01, model='nn', l2_ratio=1e-7,
        silent=True, non_linearity='relu', privacy='no_privacy', dp = 'dp', epsilon=0.5, delta=1e-5):
    train_x, train_y, test_x, test_y = dataset
        
    if hold_out_train_data != None:
        hold_out_x, hold_out_y, _, _ = hold_out_train_data

    n_in = train_x.shape[1]
    n_out = len(np.unique(train_y))

    if batch_size > len(train_y):
        batch_size = len(train_y)

    classifier = tf.estimator.Estimator(model_fn=get_model, params = [train_x.shape[0], n_in, n_hidden, n_out, non_linearity, model, privacy, dp, epsilon, delta, batch_size, learning_rate, l2_ratio, epochs])

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
