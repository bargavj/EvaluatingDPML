import os
import argparse
import pickle
import numpy as np
import tensorflow as tf

from core.classifier import train as train_model
from core.classifier import get_predictions
from core.utilities import log_loss
from core.utilities import prety_print_result
from core.utilities import get_inference_threshold
from core.utilities import generate_noise
from core.utilities import get_attribute_variations
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve

MODEL_PATH = 'model/'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


def load_attack_data():
    fname = MODEL_PATH + 'attack_train_data.npz'
    with np.load(fname) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    fname = MODEL_PATH + 'attack_test_data.npz'
    with np.load(fname) as f:
        test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x.astype('float32'), train_y.astype('int32'), test_x.astype('float32'), test_y.astype('int32')


def train_target_model(args, dataset=None, epochs=100, batch_size=100, learning_rate=0.01, clipping_threshold=1, l2_ratio=1e-7, n_hidden=50, model='nn', privacy='no_privacy', dp='dp', epsilon=0.5, delta=1e-5, save=True):
    if dataset == None:
        dataset = load_data('target_data.npz', args)
    train_x, train_y, test_x, test_y = dataset

    classifier, aux = train_model(
        dataset, 
        n_hidden=n_hidden, 
        epochs=epochs, 
        learning_rate=learning_rate, 
        clipping_threshold=clipping_threshold, 
        batch_size=batch_size, 
        model=model, 
        l2_ratio=l2_ratio, 
        silent=False, 
        privacy=privacy, 
        dp=dp, 
        epsilon=epsilon, 
        delta=delta)
    # test data for attack model
    attack_x, attack_y = [], []

    # data used in training, label is 1
    pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': train_x},
        num_epochs=1,
        shuffle=False)

    predictions = classifier.predict(input_fn=pred_input_fn)
    _, pred_scores = get_predictions(predictions)

    attack_x.append(pred_scores)
    attack_y.append(np.ones(train_x.shape[0]))
    
    # data not used in training, label is 0
    pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
        x={'x': test_x},
        num_epochs=1,
        shuffle=False)

    predictions = classifier.predict(input_fn=pred_input_fn)
    _, pred_scores = get_predictions(predictions)
    
    attack_x.append(pred_scores)
    attack_y.append(np.zeros(test_x.shape[0]))

    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')

    if save:
        np.savez(MODEL_PATH + 'attack_test_data.npz', attack_x, attack_y)

    classes = np.concatenate([train_y, test_y])
    return attack_x, attack_y, classes, classifier, aux


def train_shadow_models(args, n_hidden=50, epochs=100, n_shadow=20, learning_rate=0.05, batch_size=100, l2_ratio=1e-7, model='nn', privacy='no_privacy', dp='dp', epsilon=0.5, delta=1e-5, save=True):
    attack_x, attack_y = [], []
    classes = []
    for i in range(n_shadow):
        #print('Training shadow model {}'.format(i))
        dataset = load_data('shadow{}_data.npz'.format(i), args)
        train_x, train_y, test_x, test_y = dataset

        # train model
        classifier = train_model(
            dataset, 
            n_hidden=n_hidden, 
            epochs=epochs, 
            learning_rate=learning_rate, 
            batch_size=batch_size, 
            model=model, 
            l2_ratio=l2_ratio, 
            privacy=privacy, 
            dp=dp, 
            epsilon=epsilon, 
            delta=delta)
        #print('Gather training data for attack model')
        attack_i_x, attack_i_y = [], []

        # data used in training, label is 1
        pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={'x': train_x},
            num_epochs=1,
            shuffle=False)

        predictions = classifier.predict(input_fn=pred_input_fn)
        _, pred_scores = get_predictions(predictions)
    
        attack_i_x.append(pred_scores)
        attack_i_y.append(np.ones(train_x.shape[0]))
    
        # data not used in training, label is 0
        pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={'x': test_x},
            num_epochs=1,
            shuffle=False)

        predictions = classifier.predict(input_fn=pred_input_fn)
        _, pred_scores = get_predictions(predictions)
    
        attack_i_x.append(pred_scores)
        attack_i_y.append(np.zeros(test_x.shape[0]))
        
        attack_x += attack_i_x
        attack_y += attack_i_y
        classes.append(np.concatenate([train_y, test_y]))
    # train data for attack model
    attack_x = np.vstack(attack_x)
    attack_y = np.concatenate(attack_y)
    attack_x = attack_x.astype('float32')
    attack_y = attack_y.astype('int32')
    classes = np.concatenate(classes)

    if save:
        np.savez(MODEL_PATH + 'attack_train_data.npz', attack_x, attack_y)

    return attack_x, attack_y, classes


def train_attack_model(classes, dataset=None, n_hidden=50, learning_rate=0.01, batch_size=200, epochs=50, model='nn', l2_ratio=1e-7):
    if dataset is None:
        dataset = load_attack_data()
    train_x, train_y, test_x, test_y = dataset

    train_classes, test_classes = classes
    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(train_classes)

    pred_y = []
    shadow_membership, target_membership = [], []
    shadow_pred_scores, target_pred_scores = [], []
    shadow_class_labels, target_class_labels = [], []
    for c in unique_classes:
        #print('Training attack model for class {}...'.format(c))
        c_train_indices = train_indices[train_classes == c]
        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]
        c_test_indices = test_indices[test_classes == c]
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]
        c_dataset = (c_train_x, c_train_y, c_test_x, c_test_y)
        classifier = train_model(
            c_dataset, 
            n_hidden=n_hidden, 
            epochs=epochs, 
            learning_rate=learning_rate, 
            batch_size=batch_size, 
            model=model, 
            l2_ratio=l2_ratio)
        
        pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={'x': c_train_x},
            num_epochs=1,
            shuffle=False)
        predictions = classifier.predict(input_fn=pred_input_fn)
        c_pred_y, c_pred_scores = get_predictions(predictions)
        shadow_membership.append(c_train_y)
        shadow_pred_scores.append(c_pred_scores)
        shadow_class_labels.append([c]*len(c_train_indices))

        pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={'x': c_test_x},
            num_epochs=1,
            shuffle=False)
        predictions = classifier.predict(input_fn=pred_input_fn)
        c_pred_y, c_pred_scores = get_predictions(predictions)
        pred_y.append(c_pred_y)
        target_membership.append(c_test_y)
        target_pred_scores.append(c_pred_scores)
        target_class_labels.append([c]*len(c_test_indices))

    print('-' * 10 + 'FINAL EVALUATION' + '-' * 10 + '\n')
    pred_y = np.concatenate(pred_y)
    shadow_membership = np.concatenate(shadow_membership)
    target_membership = np.concatenate(target_membership)
    shadow_pred_scores = np.concatenate(shadow_pred_scores)
    target_pred_scores = np.concatenate(target_pred_scores)
    shadow_class_labels = np.concatenate(shadow_class_labels)
    target_class_labels = np.concatenate(target_class_labels)
    prety_print_result(target_membership, pred_y)
    fpr, tpr, thresholds = roc_curve(target_membership, pred_y, pos_label=1)
    attack_adv = tpr[1] - fpr[1]
    return (attack_adv, shadow_pred_scores, target_pred_scores, shadow_membership, target_membership, shadow_class_labels, target_class_labels)


def save_data(args):
    print('-' * 10 + 'SAVING DATA TO DISK' + '-' * 10 + '\n')
    target_size = args.target_data_size
    gamma = args.target_test_train_ratio
    DATA_PATH = 'data/' + args.train_dataset + '/'
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)

    x = pickle.load(open('../dataset/'+args.train_dataset+'_features.p', 'rb'))
    y = pickle.load(open('../dataset/'+args.train_dataset+'_labels.p', 'rb'))
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(x.shape, y.shape)

    # assert if data is enough for sampling target data
    assert(len(x) >= (1 + gamma) * target_size)
    x, train_x, y, train_y = train_test_split(x, y, test_size=target_size, stratify=y)
    print("Training set size:  X: {}, y: {}".format(train_x.shape, train_y.shape))
    x, test_x, y, test_y = train_test_split(x, y, test_size=int(gamma*target_size), stratify=y)
    print("Test set size:  X: {}, y: {}".format(test_x.shape, test_y.shape))

    # save target data
    print('Saving data for target model')
    np.savez(DATA_PATH + 'target_data.npz', train_x, train_y, test_x, test_y)

    # assert if remaining data is enough for sampling shadow data
    assert(len(x) >= (1 + gamma) * target_size)

    # save shadow data
    for i in range(args.n_shadow):
        print('Saving data for shadow model {}'.format(i))
        train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=target_size, test_size=int(gamma*target_size), stratify=y)
        print("Training set size:  X: {}, y: {}".format(train_x.shape, train_y.shape))
        print("Test set size:  X: {}, y: {}".format(test_x.shape, test_y.shape))
        np.savez(DATA_PATH + 'shadow{}_data.npz'.format(i), train_x, train_y, test_x, test_y)


def load_data(data_name, args):
    DATA_PATH = 'data/' + args.train_dataset + '/'
    target_size = args.target_data_size
    gamma = args.target_test_train_ratio
    with np.load(DATA_PATH + data_name) as f:
        train_x, train_y, test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]

    train_x = np.array(train_x, dtype=np.float32)
    test_x = np.array(test_x, dtype=np.float32)

    train_y = np.array(train_y, dtype=np.int32)
    test_y = np.array(test_y, dtype=np.int32)

    return train_x, train_y, test_x[:int(gamma*target_size)], test_y[:int(gamma*target_size)]


def shokri_membership_inference(args, attack_test_x, attack_test_y, test_classes):
    print('-' * 10 + 'SHOKRI\'S MEMBERSHIP INFERENCE' + '-' * 10 + '\n')    
    print('-' * 10 + 'TRAIN SHADOW' + '-' * 10 + '\n')
    attack_train_x, attack_train_y, train_classes = train_shadow_models(
        args=args,
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        n_shadow=args.n_shadow,
        n_hidden=args.target_n_hidden,
        l2_ratio=args.target_l2_ratio,
        model=args.target_model,
        save=args.save_model)

    print('-' * 10 + 'TRAIN ATTACK' + '-' * 10 + '\n')
    dataset = (attack_train_x, attack_train_y, attack_test_x, attack_test_y)
    return train_attack_model(
        dataset=dataset,
        epochs=args.attack_epochs,
        batch_size=args.attack_batch_size,
        learning_rate=args.attack_learning_rate,
        n_hidden=args.attack_n_hidden,
        l2_ratio=args.attack_l2_ratio,
        model=args.attack_model,
        classes=(train_classes, test_classes))


def yeom_membership_inference(per_instance_loss, membership, train_loss, test_loss=None):
    print('-' * 10 + 'YEOM\'S MEMBERSHIP INFERENCE' + '-' * 10 + '\n')    
    if test_loss == None:
    	pred_membership = np.where(per_instance_loss <= train_loss, 1, 0)
    else:
    	pred_membership = np.where(stats.norm(0, train_loss).pdf(per_instance_loss) >= stats.norm(0, test_loss).pdf(per_instance_loss), 1, 0)
    prety_print_result(membership, pred_membership)
    return pred_membership


def proposed_membership_inference(v_dataset, true_x, true_y, classifier, per_instance_loss, args):
    print('-' * 10 + 'PROPOSED MEMBERSHIP INFERENCE' + '-' * 10 + '\n')
    v_train_x, v_train_y, v_test_x, v_test_y = v_dataset
    v_true_x = np.vstack([v_train_x, v_test_x])
    v_true_y = np.concatenate([v_train_y, v_test_y])    
    v_pred_y, v_membership, v_test_classes, v_classifier, aux = train_target_model(
        args=args,
        dataset=v_dataset,
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        clipping_threshold=args.target_clipping_threshold,
        n_hidden=args.target_n_hidden,
        l2_ratio=args.target_l2_ratio,
        model=args.target_model,
        privacy=args.target_privacy,
        dp=args.target_dp,
        epsilon=args.target_epsilon,
        delta=args.target_delta,
        save=args.save_model)
    v_per_instance_loss = np.array(log_loss(v_true_y, v_pred_y))
    noise_params = (args.attack_noise_type, args.attack_noise_coverage, args.attack_noise_magnitude)
    v_merlin_ratio = get_merlin_ratio(v_true_x, v_true_y, v_classifier, v_per_instance_loss, noise_params)
    merlin_ratio = get_merlin_ratio(true_x, true_y, classifier, per_instance_loss, noise_params)
    return (true_y, v_true_y, v_membership, v_per_instance_loss, v_merlin_ratio, merlin_ratio)


def evaluate_proposed_membership_inference(per_instance_loss, membership, proposed_mi_outputs, fpr_threshold=None, per_class_thresh=False):
    true_y, v_true_y, v_membership, v_per_instance_loss, v_merlin_ratio, merlin_ratio = proposed_mi_outputs
    print('-' * 10 + 'Using Yeom\'s MI with custom threshold' + '-' * 10 + '\n')
    if per_class_thresh:
        classes = np.unique(true_y)
        pred_membership = np.zeros(len(membership))
        for c in classes:
            c_indices = np.arange(len(true_y))[true_y == c]
            v_c_indices = np.arange(len(v_true_y))[v_true_y == c]
            thresh = get_inference_threshold(-v_per_instance_loss[v_c_indices], v_membership[v_c_indices], fpr_threshold)
            pred_membership[c_indices] = np.where(per_instance_loss[c_indices] <= -thresh, 1, 0)
    else:
        thresh = get_inference_threshold(-v_per_instance_loss, v_membership, fpr_threshold)
        pred_membership = np.where(per_instance_loss <= -thresh, 1, 0)
    prety_print_result(membership, pred_membership)

    print('-' * 10 + 'Using Merlin with custom threshold' + '-' * 10 + '\n')
    if per_class_thresh:
        classes = np.unique(true_y)
        pred_membership = np.zeros(len(membership))
        for c in classes:
            c_indices = np.arange(len(true_y))[true_y == c]
            v_c_indices = np.arange(len(v_true_y))[v_true_y == c]
            thresh = get_inference_threshold(v_merlin_ratio[v_c_indices], v_membership[v_c_indices], fpr_threshold)
            pred_membership[c_indices] = np.where(merlin_ratio[c_indices] >= thresh, 1, 0)
    else:
        thresh = get_inference_threshold(v_merlin_ratio, v_membership, fpr_threshold)
        pred_membership = np.where(merlin_ratio >= thresh, 1, 0)
    prety_print_result(membership, pred_membership)


def get_merlin_ratio(true_x, true_y, classifier, per_instance_loss, noise_params, max_t=100):
    counts = np.zeros(len(true_x))
    for t in range(max_t):
        noisy_x = true_x + generate_noise(true_x.shape, true_x.dtype, noise_params)
        pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={'x': noisy_x}, 
           num_epochs=1,
            shuffle=False)
        predictions = classifier.predict(input_fn=pred_input_fn)
        _, pred_y = get_predictions(predictions)
        noisy_per_instance_loss = np.array(log_loss(true_y, pred_y))
        counts += np.where(noisy_per_instance_loss > per_instance_loss, 1, 0)
    return counts / max_t


def yeom_attribute_inference(true_x, true_y, classifier, membership, features, train_loss, test_loss=None):
    print('-' * 10 + 'YEOM\'S ATTRIBUTE INFERENCE' + '-' * 10 + '\n')
    pred_membership_all = []
    for feature in features:
        orignial_attribute = np.copy(true_x[:,feature])
        low_value, high_value, true_attribute_value = get_attribute_variations(true_x, feature)
        
        true_x[:,feature] = low_value
        pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={'x': true_x},
            num_epochs=1,
            shuffle=False)
        predictions = classifier.predict(input_fn=pred_input_fn)
        _, low_op = get_predictions(predictions)
        low_op = low_op.astype('float32')
        low_op = log_loss(true_y, low_op)
        
        true_x[:,feature] = high_value
        pred_input_fn = tf.compat.v1.estimator.inputs.numpy_input_fn(
            x={'x': true_x},
            num_epochs=1,
            shuffle=False)
        predictions = classifier.predict(input_fn=pred_input_fn)
        _, high_op = get_predictions(predictions)
        high_op = high_op.astype('float32')
        high_op = log_loss(true_y, high_op)
        
        high_prob = np.sum(true_attribute_value) / len(true_attribute_value)
        low_prob = 1 - high_prob
        
        if test_loss == None:
            pred_attribute_value = np.where(low_prob * stats.norm(0, train_loss).pdf(low_op) >= high_prob * stats.norm(0, train_loss).pdf(high_op), 0, 1)
            mask = [1]*len(pred_attribute_value)
        else:
            low_mem = np.where(stats.norm(0, train_loss).pdf(low_op) >= stats.norm(0, test_loss).pdf(low_op), 1, 0)
            high_mem = np.where(stats.norm(0, train_loss).pdf(high_op) >= stats.norm(0, test_loss).pdf(high_op), 1, 0)
            pred_attribute_value = [np.argmax([low_prob * a, high_prob * b]) for a, b in zip(low_mem, high_mem)]
            mask = [a | b for a, b in zip(low_mem, high_mem)]
        
        pred_membership = mask & (pred_attribute_value ^ true_attribute_value ^ [1]*len(pred_attribute_value))
        prety_print_result(membership, pred_membership)
        pred_membership_all.append(pred_membership)
        true_x[:,feature] = orignial_attribute
    return pred_membership_all
