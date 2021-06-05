import os
import numpy as np
import tensorflow as tf

from core.constants import rdp_noise_multiplier
from core.constants import gdp_noise_multiplier
from tensorflow_privacy.privacy.analysis.rdp_accountant import compute_rdp
from tensorflow_privacy.privacy.analysis.rdp_accountant import get_privacy_spent
from tensorflow_privacy.privacy.optimizers import dp_optimizer

LOGGING = False # enables tf.train.ProfilerHook (see use below)
LOG_DIR = 'log'
CHECKPOINT_DIR = '__temp_files'

AdamOptimizer = tf.compat.v1.train.AdamOptimizer

def get_predictions(predictions):
    pred_y, pred_scores = [], []
    val = next(predictions, None)
    while val is not None:
        pred_y.append(val['classes'])
        pred_scores.append(val['probabilities'])
        val = next(predictions, None)
    return np.array(pred_y), np.array(pred_scores)


def get_model(features, labels, mode, params):
    n, n_in, n_hidden, n_out, non_linearity, model, privacy, dp, epsilon, delta, batch_size, learning_rate, clipping_threshold, l2_ratio, epochs = params
    if model == 'nn':
        #print('Using neural network...')
        input_layer = tf.reshape(features['x'], [-1, n_in])
        h1 = tf.keras.layers.Dense(n_hidden, activation=non_linearity, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio)).apply(input_layer)
        h2 = tf.keras.layers.Dense(n_hidden, activation=non_linearity, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio)).apply(h1)
        pre_logits = tf.keras.layers.Dense(n_out, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio)).apply(h2)
        logits = tf.keras.layers.Softmax().apply(pre_logits)
    elif model == 'cnn':
        #print('Using convolution neural network...') # use only on Cifar-100
        input_layer = tf.reshape(features['x'], [-1, 32, 32, 3])
        y = tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation=non_linearity).apply(input_layer)
        y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)).apply(y)
        y = tf.keras.layers.Conv2D(64, kernel_size=(3, 3), activation=non_linearity, input_shape=[-1, 32, 32, 3]).apply(y)
        y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)).apply(y)
        y = tf.keras.layers.Conv2D(128, kernel_size=(3, 3), activation=non_linearity, input_shape=[-1, 32, 32, 3]).apply(y)
        y = tf.keras.layers.MaxPooling2D(pool_size=(2, 2)).apply(y)
        y = tf.keras.layers.Flatten().apply(y)
        y = tf.nn.dropout(y, 0.2)
        h1 = tf.keras.layers.Dense(n_hidden, activation=non_linearity, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio)).apply(y)
        h2 = tf.keras.layers.Dense(n_hidden, activation=non_linearity, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio)).apply(h1)
        pre_logits = tf.keras.layers.Dense(n_out, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio)).apply(h2)
        logits = tf.keras.layers.Softmax().apply(pre_logits)
    else:
        #print('Using softmax regression...')
        input_layer = tf.reshape(features['x'], [-1, n_in])
        logits = tf.keras.layers.Dense(n_out, activation=tf.nn.softmax, kernel_regularizer=tf.keras.regularizers.l2(l2_ratio)).apply(input_layer)
    
    predictions = {
      "classes": tf.argmax(input=logits, axis=1),
      "probabilities": logits
    }

    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode,
                                          predictions=predictions)

    vector_loss = tf.keras.losses.sparse_categorical_crossentropy(labels, logits)
    scalar_loss = tf.reduce_mean(vector_loss)

    if mode == tf.estimator.ModeKeys.TRAIN:
        
        if privacy == 'grad_pert':
            if dp == 'adv_cmp':
                sigma = np.sqrt(epochs * np.log(2.5 * epochs / delta)) * (np.sqrt(np.log(2 / delta) + 2 * epsilon) + np.sqrt(np.log(2 / delta))) / epsilon
            elif dp == 'zcdp':
                sigma = np.sqrt(epochs / 2) * (np.sqrt(np.log(1 / delta) + epsilon) + np.sqrt(np.log(1 / delta))) / epsilon
            elif dp == 'rdp':
                sigma = rdp_noise_multiplier[epochs][epsilon]
            elif dp == 'gdp':
                sigma = gdp_noise_multiplier[epochs][epsilon]
            else: # if dp == 'dp'
                sigma = epochs * np.sqrt(2 * np.log(1.25 * epochs / delta)) / epsilon
    
            optimizer = dp_optimizer.DPAdamGaussianOptimizer(
                            l2_norm_clip=clipping_threshold,
                            noise_multiplier=sigma,
                            num_microbatches=batch_size,
                            learning_rate=learning_rate,
                            ledger=None)
            opt_loss = vector_loss
        else:
            optimizer = AdamOptimizer(learning_rate=learning_rate)
            opt_loss = scalar_loss
        global_step = tf.compat.v1.train.get_global_step()
        train_op = optimizer.minimize(loss=opt_loss, global_step=global_step)
        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=scalar_loss,
                                          train_op=train_op)

    elif mode == tf.estimator.ModeKeys.EVAL:
        eval_metric_ops = {
            'accuracy':
                tf.compat.v1.metrics.accuracy(
                    labels=labels,
                     predictions=predictions["classes"])
        }

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=scalar_loss,
                                          eval_metric_ops=eval_metric_ops)


def train(dataset, n_hidden=50, batch_size=100, epochs=100, learning_rate=0.01, clipping_threshold=1, model='nn', l2_ratio=1e-7, silent=True, non_linearity='relu', privacy='no_privacy', dp = 'dp', epsilon=0.5, delta=1e-5):
    train_x, train_y, test_x, test_y = dataset

    n_in = train_x.shape[1]
    n_out = len(np.unique(train_y))

    if batch_size > len(train_y):
        batch_size = len(train_y)

    if not os.path.exists(CHECKPOINT_DIR):
        os.makedirs(CHECKPOINT_DIR)
    
    classifier = tf.estimator.Estimator(
            model_fn=get_model,
            #model_dir=CHECKPOINT_DIR,
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
                clipping_threshold,
                l2_ratio,
                epochs])

    train_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': train_x},
        y=train_y,
        batch_size=batch_size,
        num_epochs=epochs,
        shuffle=True)
    train_eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': train_x},
        y=train_y,
        num_epochs=1,
        shuffle=False)
    test_eval_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': test_x},
        y=test_y,
        num_epochs=1,
        shuffle=False)

    steps_per_epoch = train_x.shape[0] // batch_size

    if not os.path.exists(LOG_DIR):
        os.makedirs(LOG_DIR)
    for epoch in range(1, epochs + 1):
        hooks = []
        if LOGGING:
            hooks.append(tf.train.ProfilerHook(
                output_dir=LOG_DIR,
                save_steps=30))
        # This hook will save traces of what tensorflow is doing
        # during the training of each model. View the combined trace
        # by running `combine_traces.py`

        classifier.train(input_fn=train_input_fn,
                steps=steps_per_epoch,
                hooks=hooks)
    
        if not silent:
            eval_results = classifier.evaluate(input_fn=train_eval_input_fn)
            print('Train loss after %d epochs is: %.3f' % (epoch, eval_results['loss']))

    if not silent:
        eval_results = classifier.evaluate(input_fn=train_eval_input_fn)
        train_loss = eval_results['loss']
        train_acc = eval_results['accuracy']
        print('Train accuracy is: %.3f' % (train_acc))

        eval_results = classifier.evaluate(input_fn=test_eval_input_fn)
        test_loss = eval_results['loss']
        test_acc = eval_results['accuracy']
        print('Test accuracy is: %.3f' % (test_acc))

        # warning: silent flag is only used for target model training, 
        # as it also returns auxiliary information
        return classifier, (train_loss, train_acc, test_loss, test_acc)

    return classifier
