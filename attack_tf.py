from classifier_tf import train as train_model, load_dataset, get_predictions
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_curve
from scipy import stats
import numpy as np
import tensorflow as tf
import argparse
import os
import imp
import pickle
import matplotlib.pyplot as plt
import random

MODEL_PATH = './model/'
DATA_PATH = './data/'
RESULT_PATH = './results/'

SMALL_VALUE = 0.001


if not os.path.exists(MODEL_PATH):
    os.makedirs(MODEL_PATH)

if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)


def load_trained_indices():
    fname = MODEL_PATH + 'data_indices.npz'
    with np.load(fname) as f:
        indices = [f['arr_%d' % i] for i in range(len(f.files))]
    return indices


def get_data_indices(data_size, target_train_size=int(1e4), sample_target_data=True):
    train_indices = np.arange(data_size)
    if sample_target_data:
        target_data_indices = np.random.choice(train_indices, target_train_size, replace=False)
        shadow_indices = np.setdiff1d(train_indices, target_data_indices)
    else:
        target_data_indices = train_indices[:target_train_size]
        shadow_indices = train_indices[target_train_size:]
    return target_data_indices, shadow_indices


def load_attack_data():
    fname = MODEL_PATH + 'attack_train_data.npz'
    with np.load(fname) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    fname = MODEL_PATH + 'attack_test_data.npz'
    with np.load(fname) as f:
        test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x.astype('float32'), train_y.astype('int32'), test_x.astype('float32'), test_y.astype('int32')


def train_target_model(dataset, epochs=100, batch_size=100, learning_rate=0.01, l2_ratio=1e-7,
                       n_hidden=50, model='nn', save=True, privacy='no_privacy', dp='dp', epsilon=0.5, delta=1e-5):
    train_x, train_y, test_x, test_y = dataset

    classifier, _, _, train_loss, train_acc, test_acc = train_model(dataset, n_hidden=n_hidden, epochs=epochs, learning_rate=learning_rate,
                               batch_size=batch_size, model=model, l2_ratio=l2_ratio, silent=False, privacy=privacy, dp=dp, epsilon=epsilon, delta=delta)
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
        np.savez(MODEL_PATH + 'target_model.npz', *lasagne.layers.get_all_param_values(output_layer))

    classes = np.concatenate([train_y, test_y])
    return attack_x, attack_y, classes, train_loss, classifier, train_acc, test_acc


def train_shadow_models(n_hidden=50, epochs=100, n_shadow=20, learning_rate=0.05, batch_size=100, l2_ratio=1e-7,
                        model='nn', save=True):
    # for attack model
    attack_x, attack_y = [], []
    classes = []
    for i in range(n_shadow):
        #print('Training shadow model {}'.format(i))
        data = load_data('shadow{}_data.npz'.format(i))
        train_x, train_y, test_x, test_y = data

        # train model
        classifier, _, _, _, _, _ = train_model(data, n_hidden=n_hidden, epochs=epochs, learning_rate=learning_rate,
                                   batch_size=batch_size, model=model, l2_ratio=l2_ratio)
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


def train_attack_model(classes, dataset=None, n_hidden=50, learning_rate=0.01, batch_size=200, epochs=50,
                       model='nn', l2_ratio=1e-7):
    if dataset is None:
        dataset = load_attack_data()

    train_x, train_y, test_x, test_y = dataset

    train_classes, test_classes = classes
    train_indices = np.arange(len(train_x))
    test_indices = np.arange(len(test_x))
    unique_classes = np.unique(train_classes)

    true_y = []
    pred_y = []
    pred_scores = []
    true_x = []
    for c in unique_classes:
        #print('Training attack model for class {}...'.format(c))
        c_train_indices = train_indices[train_classes == c]
        c_train_x, c_train_y = train_x[c_train_indices], train_y[c_train_indices]
        c_test_indices = test_indices[test_classes == c]
        c_test_x, c_test_y = test_x[c_test_indices], test_y[c_test_indices]
        c_dataset = (c_train_x, c_train_y, c_test_x, c_test_y)
        _, c_pred_y, c_pred_scores, _, _, _ = train_model(c_dataset, n_hidden=n_hidden, epochs=epochs, learning_rate=learning_rate,
                               batch_size=batch_size, model=model, l2_ratio=l2_ratio, non_linearity='relu')
        true_y.append(c_test_y)
        pred_y.append(c_pred_y)
        true_x.append(c_test_x)
        pred_scores.append(c_pred_scores)

    print('-' * 10 + 'FINAL EVALUATION' + '-' * 10 + '\n')
    true_y = np.concatenate(true_y)
    pred_y = np.concatenate(pred_y)
    true_x = np.concatenate(true_x)
    pred_scores = np.concatenate(pred_scores)
    #print('Testing Accuracy: {}'.format(accuracy_score(true_y, pred_y)))
    #print(classification_report(true_y, pred_y))
    fpr, tpr, thresholds = roc_curve(true_y, pred_y, pos_label=1)
    print(fpr, tpr, tpr-fpr)
    attack_adv = tpr[1]-fpr[1]
    #plt.plot(fpr, tpr)

    # membership
    fpr, tpr, thresholds = roc_curve(true_y, pred_scores[:,1], pos_label=1)
    #plt.plot(fpr, tpr)
    # non-membership
    fpr, tpr, thresholds = roc_curve(true_y, pred_scores[:,0], pos_label=0)
    #plt.show()

    #FILE_SUFFIX = '_nn_rdp_1000_0.npz'
    #np.save(DATA_PATH + 'true_x' + FILE_SUFFIX, true_x)
    #np.save(DATA_PATH + 'true_y' + FILE_SUFFIX, true_y)
    #np.save(DATA_PATH + 'pred_scores' + FILE_SUFFIX, pred_scores)
    return attack_adv, pred_scores


def save_data():
    print('-' * 10 + 'SAVING DATA TO DISK' + '-' * 10 + '\n')

    #x, y, test_x, test_y = load_dataset(args.train_feat, args.train_label, args.test_feat, args.test_label) # changed train_label to test_label
    x = pickle.load(open('dataset/'+args.train_dataset+'_features.p', 'rb'))
    y = pickle.load(open('dataset/'+args.train_dataset+'_labels.p', 'rb'))
    x, y = np.matrix(x), np.array(y)

    test_x, test_y = None, None
    y = y.astype('int32')
    if test_x is None:
        print('Splitting train/test data with ratio {}/{}'.format(1 - args.test_ratio, args.test_ratio))
        x, test_x, y, test_y = train_test_split(x, y, test_size=args.test_ratio, stratify=y)

    # need to partition target and shadow model data
    assert len(x) > 2 * args.target_data_size

    target_data_indices, shadow_indices = get_data_indices(len(x), target_train_size=args.target_data_size)
    np.savez(MODEL_PATH + 'data_indices.npz', target_data_indices, shadow_indices)

    # target model's data
    print('Saving data for target model')
    train_x, train_y = x[target_data_indices], y[target_data_indices]
    size = len(target_data_indices)
    if size < len(test_x):
        test_x = test_x[:size]
        test_y = test_y[:size]
    # save target data
    np.savez(DATA_PATH + 'target_data.npz', train_x, train_y, test_x, test_y)

    # shadow model's data
    target_size = len(target_data_indices)
    shadow_x, shadow_y = x[shadow_indices], y[shadow_indices]
    shadow_indices = np.arange(len(shadow_indices))

    for i in range(args.n_shadow):
        print('Saving data for shadow model {}'.format(i))
        shadow_i_indices = np.random.choice(shadow_indices, 2 * target_size, replace=False)
        shadow_i_x, shadow_i_y = shadow_x[shadow_i_indices], shadow_y[shadow_i_indices]
        train_x, train_y = shadow_i_x[:target_size], shadow_i_y[:target_size]
        test_x, test_y = shadow_i_x[target_size:], shadow_i_y[target_size:]
        np.savez(DATA_PATH + 'shadow{}_data.npz'.format(i), train_x, train_y, test_x, test_y)


def load_data(data_name):
    with np.load(DATA_PATH + data_name) as f:
        train_x, train_y, test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]

    train_x = np.array(train_x, dtype=np.float32)
    test_x = np.array(test_x, dtype=np.float32)

    train_y = np.array(train_y, dtype=np.int32)
    test_y = np.array(test_y, dtype=np.int32)

    return train_x, train_y, test_x, test_y


def attack_experiment(attack_test_x, attack_test_y, test_classes,):
    print('-' * 10 + 'TRAIN SHADOW' + '-' * 10 + '\n')
    attack_train_x, attack_train_y, train_classes = train_shadow_models(
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


def membership_inference(true_y, pred_y, membership, train_loss):
	print('-' * 10 + 'MEMBERSHIP INFERENCE' + '-' * 10 + '\n')    
	pred_membership = np.where(log_loss(true_y, pred_y) <= train_loss, 1, 0)
	#print(classification_report(membership, pred_membership))
	fpr, tpr, thresholds = roc_curve(membership, pred_membership, pos_label=1)
	print(fpr, tpr, tpr-fpr)
	mem_adv = tpr[1]-fpr[1]
	#plt.plot(fpr, tpr)

	# membership
	fpr, tpr, thresholds = roc_curve(membership, max(log_loss(true_y, pred_y)) - log_loss(true_y, pred_y), pos_label=1)
	#plt.plot(fpr, tpr)
	# non-membership
	fpr, tpr, thresholds = roc_curve(membership, log_loss(true_y, pred_y), pos_label=0)
	#plt.show()
	return mem_adv, log_loss(true_y, pred_y)


def attribute_inference(true_x, true_y, batch_size, classifier, train_loss, features):
    print('-' * 10 + 'ATTRIBUTE INFERENCE' + '-' * 10 + '\n')
    attr_adv, attr_mem, attr_pred = [], [], []
    for feature in features:
        low_op, high_op = [], []

        low_data, high_data, membership = getAttributeVariations(true_x, feature)

        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': low_data},
            num_epochs=1,
            shuffle=False)

        predictions = classifier.predict(input_fn=pred_input_fn)
        _, low_op = get_predictions(predictions)
        
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': high_data},
            num_epochs=1,
            shuffle=False)

        predictions = classifier.predict(input_fn=pred_input_fn)
        _, high_op = get_predictions(predictions)

        low_op = low_op.astype('float32')
        high_op = high_op.astype('float32')

        low_op = log_loss(true_y, low_op)
        high_op = log_loss(true_y, high_op)
	    
        pred_membership = np.where(stats.norm(0, train_loss).pdf(low_op) >= stats.norm(0, train_loss).pdf(high_op), 0, 1)
        fpr, tpr, thresholds = roc_curve(membership, pred_membership, pos_label=1)
        print(fpr, tpr, tpr-fpr)
        attr_adv.append(tpr[1]-fpr[1])
		#plt.plot(fpr, tpr)

		# membership
        fpr, tpr, thresholds = roc_curve(membership, stats.norm(0, train_loss).pdf(high_op) - stats.norm(0, train_loss).pdf(low_op), pos_label=1)
		#plt.plot(fpr, tpr)
		# non-membership
        fpr, tpr, thresholds = roc_curve(membership, stats.norm(0, train_loss).pdf(low_op) - stats.norm(0, train_loss).pdf(high_op), pos_label=0)
		#plt.show()

		
        attr_mem.append(membership)
        attr_pred.append(np.vstack((low_op, high_op)))
    return attr_adv, attr_mem, attr_pred


def getAttributeVariations(data, feature):
	low_data, high_data = np.copy(data), np.copy(data)
	pivot = np.quantile(data[:,feature], 0.5)
	low = np.quantile(data[:,feature], 0.25)
	high = np.quantile(data[:,feature], 0.75)
	membership = np.where(data[:,feature] <= pivot, 0, 1)
	low_data[:,feature] = low
	high_data[:,feature] = high
	return low_data, high_data, membership


def log_loss(a, b):
	return [-np.log(max(b[i,a[i]], SMALL_VALUE)) for i in range(len(a))]


def get_random_features(data, pool, size):
    random.seed(21312)
    features = set()
    while(len(features) < size):
        feat = random.choice(pool)
        if len(np.unique(data[:,feat])) > 1:
            features.add(feat)
    return list(features)


def run_experiment():
    print('-' * 10 + 'TRAIN TARGET' + '-' * 10 + '\n')
    dataset = load_data('target_data.npz')
    train_x, train_y, test_x, test_y = dataset
    true_x = np.vstack((train_x, test_x))
    true_y = np.append(train_y, test_y)
    batch_size = args.target_batch_size
    pred_y, membership, test_classes, train_loss, classifier, train_acc, test_acc = train_target_model(
        dataset=dataset,
        epochs=args.target_epochs,
        batch_size=args.target_batch_size,
        learning_rate=args.target_learning_rate,
        n_hidden=args.target_n_hidden,
        l2_ratio=args.target_l2_ratio,
        model=args.target_model,
        privacy=args.target_privacy,
        dp=args.target_dp,
        epsilon=args.target_epsilon,
        delta=args.target_delta,
        save=args.save_model)
    
    features = get_random_features(true_x, range(true_x.shape[1]), 5)
    print(features)
    attack_adv, attack_pred = attack_experiment(pred_y, membership, test_classes)
    mem_adv, mem_pred = membership_inference(true_y, pred_y, membership, train_loss)
    attr_adv, attr_mem, attr_pred = attribute_inference(true_x, true_y, batch_size, classifier, train_loss, features)

    if not os.path.exists(RESULT_PATH+args.train_dataset):
    	os.makedirs(RESULT_PATH+args.train_dataset)

    pickle.dump([train_acc, test_acc, train_loss, membership, attack_adv, attack_pred, mem_adv, mem_pred, attr_adv, attr_mem, attr_pred, features], open(RESULT_PATH+args.train_dataset+'/'+args.target_model+'_'+args.target_privacy+'_'+args.target_dp+'_'+str(args.target_epsilon)+'_'+str(args.run)+'.p', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dataset', type=str)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--test_feat', type=str, default=None)
    parser.add_argument('--test_label', type=str, default=None)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--save_data', type=int, default=0)
    # if test not give, train test split configuration
    parser.add_argument('--test_ratio', type=float, default=0.2)
    # target and shadow model configuration
    parser.add_argument('--n_shadow', type=int, default=5)
    parser.add_argument('--target_data_size', type=int, default=int(1e4))
    parser.add_argument('--target_model', type=str, default='nn')
    parser.add_argument('--target_learning_rate', type=float, default=0.01)
    parser.add_argument('--target_batch_size', type=int, default=200)
    parser.add_argument('--target_n_hidden', type=int, default=256)
    parser.add_argument('--target_epochs', type=int, default=100)
    parser.add_argument('--target_l2_ratio', type=float, default=1e-8)
    parser.add_argument('--target_privacy', type=str, default='no_privacy')
    parser.add_argument('--target_dp', type=str, default='dp')
    parser.add_argument('--target_epsilon', type=float, default=0.5)
    parser.add_argument('--target_delta', type=float, default=1e-5)
    # attack model configuration
    parser.add_argument('--attack_model', type=str, default='nn')
    parser.add_argument('--attack_learning_rate', type=float, default=0.01)
    parser.add_argument('--attack_batch_size', type=int, default=100)
    parser.add_argument('--attack_n_hidden', type=int, default=64)
    parser.add_argument('--attack_epochs', type=int, default=100)
    parser.add_argument('--attack_l2_ratio', type=float, default=1e-6)

    # parse configuration
    args = parser.parse_args()
    print(vars(args))
    if args.save_data:
        save_data()
    else:
    	run_experiment()
