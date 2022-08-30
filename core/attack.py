import os
import pickle
import numpy as np
import tensorflow.compat.v1 as tf

from core.classifier import train as train_model
from core.classifier import get_predictions
from core.utilities import log_loss
from core.utilities import pretty_print_result
from core.utilities import get_inference_threshold
from core.utilities import generate_noise
from core.utilities import get_attribute_variations
from core.utilities import plot_layer_outputs
from core.data_util import load_attack_data
from core.data_util import save_data
from core.data_util import load_data
from scipy.stats import norm
from sklearn.metrics import roc_curve
from sklearn.preprocessing import QuantileTransformer

MODEL_PATH = 'model/'

if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)


def train_target_model(args, dataset=None, epochs=100, batch_size=100, learning_rate=0.01, clipping_threshold=1, l2_ratio=1e-7, n_hidden=50, n_out=None, model='nn', privacy='no_privacy', dp='dp', epsilon=0.5, delta=1e-5, save=True):
    """
    Wrapper function that trains the target model over the sensitive data.
    """
    if dataset == None:
        dataset = load_data('target_data.npz', args)
    train_x, train_y, test_x, test_y = dataset

    classifier, aux = train_model(
        dataset, 
        n_out=n_out, 
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
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
        x={'x': train_x},
        num_epochs=1,
        shuffle=False)

    predictions = classifier.predict(input_fn=pred_input_fn)
    _, pred_scores = get_predictions(predictions)

    attack_x.append(pred_scores)
    attack_y.append(np.ones(train_x.shape[0]))
    
    # data not used in training, label is 0
    pred_input_fn = tf.estimator.inputs.numpy_input_fn(
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
    """
    Wrapper function to train the shadow models similar to the target model.
    Shadow model training is peformed over the hold-out data.
    """
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
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': train_x},
            num_epochs=1,
            shuffle=False)

        predictions = classifier.predict(input_fn=pred_input_fn)
        _, pred_scores = get_predictions(predictions)
    
        attack_i_x.append(pred_scores)
        attack_i_y.append(np.ones(train_x.shape[0]))
    
        # data not used in training, label is 0
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
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
    """
    Wrapper function to train the meta-model over the shadow models' output.
    During inference time, the meta-model takes the target model's output and 
    predicts if a query record is part of the target model's training set.
    """
    if dataset is None:
        dataset = load_attack_data(MODEL_PATH)
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
        
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': c_train_x},
            num_epochs=1,
            shuffle=False)
        predictions = classifier.predict(input_fn=pred_input_fn)
        c_pred_y, c_pred_scores = get_predictions(predictions)
        shadow_membership.append(c_train_y)
        shadow_pred_scores.append(c_pred_scores)
        shadow_class_labels.append([c]*len(c_train_indices))

        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
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


def shokri_membership_inference(args, attack_test_x, attack_test_y, test_classes):
    """
    Wrapper function for Shokri et al. membership inference attack 
    that trains the shadow models and the meta-model.
    """
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
    """
    Yeom et al. membership inference attack that uses the 
    per-instance loss to predict the record membership.
    """
    print('-' * 10 + 'YEOM\'S MEMBERSHIP INFERENCE' + '-' * 10 + '\n')    
    if test_loss == None:
        pred_membership = np.where(per_instance_loss <= train_loss, 1, 0)
    else:
        pred_membership = np.where(norm(0, train_loss).pdf(per_instance_loss) >= norm(0, test_loss).pdf(per_instance_loss), 1, 0)
    #pretty_print_result(membership, pred_membership)
    return pred_membership


def proposed_membership_inference(v_dataset, true_x, true_y, classifier, per_instance_loss, args):
    """
    Our proposed membership inference attacks that use threshold-selection 
    procedure for Yeom and Merlin attacks. The function returns the 
    per-instance loss and merlin-ratio over target and hold-out sets.
    """
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
    """
    Evaluates the Yeom and Merlin attacks for a given FPR threshold.
    """
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
    pretty_print_result(membership, pred_membership)

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
    pretty_print_result(membership, pred_membership)


def get_merlin_ratio(true_x, true_y, classifier, per_instance_loss, noise_params, max_t=100):
    """
    Returns the merlin-ratio for the Merlin attack, the merlin-ratio 
    is between 0 and 1.
    """
    counts = np.zeros(len(true_x))
    for t in range(max_t):
        noisy_x = true_x + generate_noise(true_x.shape, true_x.dtype, noise_params)
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': noisy_x}, 
           num_epochs=1,
            shuffle=False)
        predictions = classifier.predict(input_fn=pred_input_fn)
        _, pred_y = get_predictions(predictions)
        noisy_per_instance_loss = np.array(log_loss(true_y, pred_y))
        counts += np.where(noisy_per_instance_loss > per_instance_loss, 1, 0)
    return counts / max_t


def yeom_attribute_inference(true_x, true_y, classifier, membership, features, train_loss, test_loss=None):
    """
    Yeom et al.'s attribute inference attack for binary attributes.
    """
    print('-' * 10 + 'YEOM\'S ATTRIBUTE INFERENCE' + '-' * 10 + '\n')
    pred_membership_all = []
    for feature in features:
        orignial_attribute = np.copy(true_x[:,feature])
        low_value, high_value, true_attribute_value = get_attribute_variations(true_x, feature)
        
        true_x[:,feature] = low_value
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': true_x},
            num_epochs=1,
            shuffle=False)
        predictions = classifier.predict(input_fn=pred_input_fn)
        _, low_op = get_predictions(predictions)
        low_op = low_op.astype('float32')
        low_op = log_loss(true_y, low_op)
        
        true_x[:,feature] = high_value
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
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
            pred_attribute_value = np.where(low_prob * norm(0, train_loss).pdf(low_op) >= high_prob * norm(0, train_loss).pdf(high_op), 0, 1)
            mask = [1]*len(pred_attribute_value)
        else:
            low_mem = np.where(norm(0, train_loss).pdf(low_op) >= norm(0, test_loss).pdf(low_op), 1, 0)
            high_mem = np.where(norm(0, train_loss).pdf(high_op) >= norm(0, test_loss).pdf(high_op), 1, 0)
            pred_attribute_value = [np.argmax([low_prob * a, high_prob * b]) for a, b in zip(low_mem, high_mem)]
            mask = [a | b for a, b in zip(low_mem, high_mem)]
        
        pred_membership = mask & (pred_attribute_value ^ true_attribute_value ^ [1]*len(pred_attribute_value))
        pretty_print_result(membership, pred_membership)
        pred_membership_all.append(pred_membership)
        true_x[:,feature] = orignial_attribute
    return pred_membership_all

        
def nematode(rows, cols):
    """
    Adds small noise to avoid singular matrices.
    """
    return 1e-8 * np.random.rand(rows, cols)

    
def get_informative_neurons(pos, neg, k):
    """
    Function to find the most informative neurons of a 
    neural network model that are most correlated to the 
    sensitive attirbute value.
    """
    informative_neurons = []
    correlation_vals = []
    pos_ = pos + nematode(pos.shape[0], pos.shape[1])
    neg_ = neg + nematode(neg.shape[0], neg.shape[1])
    neuron_corr = np.corrcoef(np.hstack((np.concatenate((pos_, neg_), axis=0), np.concatenate((np.ones(len(pos)), np.zeros(len(neg))), axis=0).reshape((-1,1)))), rowvar=False)[-1, :-1]
    top_corr = sorted(list(zip(neuron_corr, list(range(len(neuron_corr))))), key=(lambda v: v[0]), reverse=True)
    
    sorted_neurons = [v[1] for v in top_corr]
    sorted_corr_vals = [v[0] for v in top_corr]
    qt = QuantileTransformer(random_state=0)
    qt.fit(np.vstack((pos[:, sorted_neurons], neg[:, sorted_neurons])))
    pos_mean = np.mean(qt.transform(pos[:, sorted_neurons]), axis=0)
    pos_std = np.std(qt.transform(pos[:, sorted_neurons]), axis=0)
    neg_mean = np.mean(qt.transform(neg[:, sorted_neurons]), axis=0)
    neg_std = np.std(qt.transform(neg[:, sorted_neurons]), axis=0)
    pickle.dump([sorted_neurons, sorted_corr_vals, (pos_mean, pos_std, neg_mean, neg_std)], open('__temp_files/neuron_plot_info.p', 'wb'))
    
    top_corr = top_corr[:k]
    print("Top %d correlations are in [%.2f, %.2f]" % (k, top_corr[k-1][0], top_corr[0][0]))
    print("\nTop %d correlated neurons:" % (k))
    for corr_val, neuron in top_corr:
        print("Neuron #%d with correlation value of %.2f" % (neuron, corr_val))
        informative_neurons.append(neuron)
        correlation_vals.append(corr_val)
    qt = QuantileTransformer(random_state=0)
    qt.fit(np.vstack((pos[:, informative_neurons], neg[:, informative_neurons])))
    pos_mean = np.mean(qt.transform(pos[:, informative_neurons]), axis=0)
    pos_std = np.std(qt.transform(pos[:, informative_neurons]), axis=0)
    neg_mean = np.mean(qt.transform(neg[:, informative_neurons]), axis=0)
    neg_std = np.std(qt.transform(neg[:, informative_neurons]), axis=0)
    return informative_neurons, correlation_vals, (pos_mean, pos_std, neg_mean, neg_std)


def get_whitebox_score(X, correlation_vals, adv_known_idx):
    """
    Returns the white-box attribute inference attack's score 
    that is strictly between 0 and 1, that indicates the 
    attack's confidence in predicting the sensitive attribute 
    value for a query record.
    """
    # robust transformation to map neuron outputs to [0,1] range
    qt = QuantileTransformer(random_state=0)
    qt.fit(X[adv_known_idx])
    return np.average(qt.transform(X), weights=correlation_vals, axis=1)


def whitebox_attack(layer_outputs, is_sensitive, adv_known_idx):
    """
    Our white-box attack that uses the neuron outputs of a 
    neural netowork model to predict the sensitive attribute 
    value for a query record.
    """
    pos_ind = list(filter(lambda x: is_sensitive[x], adv_known_idx))
    neg_ind = list(set(adv_known_idx) - set(pos_ind))
    informative_neurons, correlation_vals, plot_info = get_informative_neurons(layer_outputs[pos_ind], layer_outputs[neg_ind], k=100)
    whitebox_info_k_1 = get_whitebox_score(layer_outputs[:, informative_neurons[:1]], correlation_vals[:1], adv_known_idx)
    whitebox_info_k_10 = get_whitebox_score(layer_outputs[:, informative_neurons[:10]], correlation_vals[:10], adv_known_idx)
    whitebox_info_k_100 = get_whitebox_score(layer_outputs[:, informative_neurons], correlation_vals, adv_known_idx)
    return (whitebox_info_k_1, whitebox_info_k_10, whitebox_info_k_100), informative_neurons, correlation_vals, plot_info
