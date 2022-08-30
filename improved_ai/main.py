import sys
import os
import argparse
import pickle
import numpy as np
import tensorflow.compat.v1 as tf

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from core.data_util import sample_noniid_data
from core.data_util import load_data
from core.data_util import get_sensitive_features
from core.data_util import process_features
from core.data_util import threat_model
from core.data_util import subsample
from core.utilities import imputation_training
from core.attack import train_target_model
from core.attack import whitebox_attack
from core.attack import yeom_membership_inference
from core.classifier import get_predictions
from core.classifier import get_layer_outputs
from collections import Counter

RESULT_PATH = 'results/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)


def get_model_conf(target_classifier, true_x, true_y, target_attr, labels, attribute_dict, max_attr_vals, col_flags, skip_corr=False):
    model_conf = np.zeros((len(true_x), len(labels)))
    for val in labels:
        features = np.copy(true_x)
        features[:, target_attr] = val / max_attr_vals[target_attr]
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': process_features(features, args.train_dataset, attribute_dict, max_attr_vals, target_attr, col_flags, skip_corr=skip_corr)},
            num_epochs=1,
            shuffle=False)
        _, pred_scores = get_predictions(target_classifier.predict(input_fn=pred_input_fn))
        model_conf[:, val] = [pred_scores[i, true_y[i]] for i in range(len(true_y))]
    return model_conf


# Confidence Score-based Model Inversion Attack from Mehnaz et al. (2022)
def get_csmia_pred(target_classifier, true_x, true_y, target_attr, labels, attribute_dict, max_attr_vals, col_flags, skip_corr=False):
    pred_conf = np.zeros((len(true_x), len(labels)))
    pred_label = np.zeros((len(true_x), len(labels)))
    csmia_pred = np.zeros(len(true_x))
    for val in labels:
        features = np.copy(true_x)
        features[:, target_attr] = val / max_attr_vals[target_attr]
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': process_features(features, args.train_dataset, attribute_dict, max_attr_vals, target_attr, col_flags, skip_corr=skip_corr)},
            num_epochs=1,
            shuffle=False)
        pred_lab, pred_scores = get_predictions(target_classifier.predict(input_fn=pred_input_fn))
        pred_label[:, val] = pred_lab
        pred_conf[:, val] = np.max(pred_scores, axis=1)
    true_y = true_y.reshape((-1, 1))
    matches = np.sum(pred_label == true_y, axis=1)
    print('Records with all incorrect predictions:\t %d (%.2f%%)' % (sum(matches==0), 100*sum(matches==0)/len(matches)))
    print('Records with exactly one correct prediction:\t %d (%.2f%%)' % (sum(matches==1), 100*sum(matches==1)/len(matches)))
    print('Records with multiple correct predictions:\t %d (%.2f%%)' % (sum(matches>=2), 100*sum(matches>=2)/len(matches)))
    csmia_pred[matches==0] = np.argmin(pred_conf[matches==0], axis=1)
    csmia_pred[matches==1] = np.argmax(pred_label[matches==1] == true_y[matches==1], axis=1)
    csmia_pred[matches>=2] = np.argmax((pred_label[matches>=2] == true_y[matches>=2]) * pred_conf[matches>=2], axis=1)
    return csmia_pred


def get_wb_info(layer_outputs, adv_known_idx, labels, sensitive_test, c_idx=None):
    whitebox_info_k_1 = np.zeros((len(sensitive_test), len(labels)))
    whitebox_info_k_10 = np.zeros((len(sensitive_test), len(labels)))
    whitebox_info_k_100 = np.zeros((len(sensitive_test), len(labels)))
    plot_info_dict = {}
    for val in labels:
        if c_idx != None:
            # high adversarial knowledge setting: running k-out-of-n test (k is set to 100)
            chunks = [c_idx[i:i+100] for i in range(0, len(c_idx), 100)]
            for chunk in chunks:
                whitebox_info, informative_neurons, correlation_vals, plot_info = whitebox_attack(layer_outputs[val], sensitive_test==val, list(set(adv_known_idx) - set(chunk)))
                whitebox_info_k_1[chunk, val], whitebox_info_k_10[chunk, val], whitebox_info_k_100[chunk, val] = [v[chunk] for v in whitebox_info]
                if str(val) not in plot_info_dict:
                    whitebox_info_k_1[:, val], whitebox_info_k_10[:, val], whitebox_info_k_100[:, val] = whitebox_info
                    plot_info_dict[str(val)] = {}
                    plot_info_dict[str(val)]['informative_neurons'] = informative_neurons
                    plot_info_dict[str(val)]['correlation_vals'] = correlation_vals
                    plot_info_dict[str(val)]['plot_info'] = plot_info
        else:
            whitebox_info, informative_neurons, correlation_vals, plot_info = whitebox_attack(layer_outputs[val], sensitive_test==val, adv_known_idx)
            whitebox_info_k_1[:, val], whitebox_info_k_10[:, val], whitebox_info_k_100[:, val] = whitebox_info
            plot_info_dict[str(val)] = {}
            plot_info_dict[str(val)]['informative_neurons'] = informative_neurons
            plot_info_dict[str(val)]['correlation_vals'] = correlation_vals
            plot_info_dict[str(val)]['plot_info'] = plot_info
    return (whitebox_info_k_1, whitebox_info_k_10, whitebox_info_k_100), plot_info_dict


def get_whitebox_info(target_classifier, true_x, adv_known_idx, target_attr, labels, attribute_dict, max_attr_vals, col_flags, sensitive_test, c_idx=None, skip_corr=False):
    layer_outputs = []
    for val in labels:
        features = np.copy(true_x)
        features[:, target_attr] = val / max_attr_vals[target_attr]
        pred_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={'x': process_features(features, args.train_dataset, attribute_dict, max_attr_vals, target_attr, col_flags, skip_corr=skip_corr)},
            num_epochs=1,
            shuffle=False)
        layer_outputs.append(get_layer_outputs(target_classifier.predict(input_fn=pred_input_fn)))
    return get_wb_info(layer_outputs, adv_known_idx, labels, sensitive_test, c_idx)


def run_experiment(args):
    if not os.path.exists(RESULT_PATH + args.train_dataset):
        os.makedirs(RESULT_PATH + args.train_dataset)
    
    MODEL = str(args.skew_attribute) + '_' + str(args.skew_outcome) + '_' + str(args.target_test_train_ratio) + '_' + str(args.target_model) + '_'
    
    train_x, train_y, test_x, test_y = load_data('target_data.npz', args)
    h_train_x, h_train_y, h_test_x, h_test_y = load_data('holdout_data.npz', args)
    sk_train_x, sk_train_y, sk_test_x, sk_test_y = load_data('skewed_data.npz', args)
    sk2_train_x, sk2_train_y, sk2_test_x, sk2_test_y = load_data('skewed_2_data.npz', args)
    true_x = np.vstack((train_x, test_x, h_train_x, h_test_x, sk_train_x, sk_test_x, sk2_train_x, sk2_test_x))
    true_y = np.concatenate((train_y, test_y, h_train_y, h_test_y, sk_train_y, sk_test_y, sk2_train_y, sk2_test_y))
    
    c_size = args.candidate_size
    train_c_idx, test_c_idx, h_test_idx, sk_test_idx, sk2_test_idx, adv_known_idxs = threat_model(args, len(true_x))
    
    assert(args.attribute < 3)
    target_attrs, attribute_dict, max_attr_vals, col_flags = get_sensitive_features(args.train_dataset, train_x)
    target_attr = target_attrs[args.attribute]
    labels = [0, 1] if attribute_dict == None else list(attribute_dict[target_attr].keys())
    
    # removes sensitive records from model training
    if args.banished_records:
        banished_records_idx = pickle.load(open('data/' + args.train_dataset + '/' + str(args.attribute) + '_' + 'med' + '_' + 'banished_records.p', 'rb'))
        assert(args.run < len(banished_records_idx))
        train_idx = list(set(range(len(train_x))) - set(np.array(train_c_idx)[banished_records_idx[args.run]]))
    else:
        train_idx = range(len(train_x))
    
    # training the target model
    _, _, _, target_classifier, model_aux = train_target_model(
        args=args,
        dataset=[process_features(train_x[train_idx], args.train_dataset, attribute_dict, max_attr_vals, target_attr, col_flags, skip_corr=args.skip_corr), train_y[train_idx], process_features(test_x, args.train_dataset, attribute_dict, max_attr_vals, target_attr, col_flags, skip_corr=args.skip_corr), test_y],
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
    train_loss, train_acc, test_loss, test_acc = model_aux
    
    model_conf = get_model_conf(target_classifier, true_x, true_y, target_attr, labels, attribute_dict, max_attr_vals, col_flags, skip_corr=args.skip_corr)
    yeom_pred = yeom_membership_inference(-np.log(model_conf), None, train_loss)
    yeom_pred_2 = yeom_membership_inference(-np.log(model_conf), None, train_loss, test_loss)
    csmia_pred = get_csmia_pred(target_classifier, true_x, true_y, target_attr, labels, attribute_dict, max_attr_vals, col_flags, skip_corr=args.skip_corr)
    
    sensitive_test = true_x[:, target_attr] * max_attr_vals[target_attr]
    known_test = process_features(true_x, args.train_dataset, attribute_dict, max_attr_vals, target_attr, col_flags, skip_sensitive=True, skip_corr=args.skip_corr)
    
    prior_prob = np.zeros(len(labels))
    for k, v in Counter(sensitive_test).items():
        prior_prob[int(k)] = v / len(sensitive_test)
    
    for threat_level in ['low', 'low2', 'med', 'high']:
        print("\nAdversary's knowledge of data: {}".format(threat_level))
        adv_known_idx_ = adv_known_idxs[threat_level]
        
        for sample_size in [50, 500, 5000, 50000]:
            print("\nAdversary's data sample size: {}".format(sample_size))
            adv_known_idx = subsample(adv_known_idx_, true_x[adv_known_idx_, target_attr] * max_attr_vals[target_attr], sample_size)
            sensitive_train = true_x[adv_known_idx, target_attr] * max_attr_vals[target_attr]
            known_train = process_features(true_x[adv_known_idx], args.train_dataset, attribute_dict, max_attr_vals, target_attr, col_flags, skip_sensitive=True, skip_corr=args.skip_corr)
            
            # training the imputation model
            imp_conf, imp_aux = imputation_training(args, known_train, sensitive_train, known_test, sensitive_test, clf_type='nn', epochs=10)
            
            print('\n\tTrain\tTest')
            print('PMC:\t%.2f\t%.2f' % (max(Counter(sensitive_test[train_c_idx]).values()) / c_size, max(Counter(sensitive_test[test_c_idx]).values()) / c_size))
            print('IP:\t%.2f\t%.2f' % (sum(sensitive_test[train_c_idx] == np.argmax(imp_conf[train_c_idx], axis=1)) / c_size, sum(sensitive_test[test_c_idx] == np.argmax(imp_conf[test_c_idx], axis=1)) / c_size))
            print('Yeom1:\t%.2f\t%.2f' % (sum(sensitive_test[train_c_idx] == np.argmax(yeom_pred[train_c_idx] * prior_prob, axis=1)) / c_size, sum(sensitive_test[test_c_idx] == np.argmax(yeom_pred[test_c_idx] * prior_prob, axis=1)) / c_size))
            print('Yeom2:\t%.2f\t%.2f' % (sum(sensitive_test[train_c_idx] == np.argmax(yeom_pred_2[train_c_idx] * prior_prob, axis=1)) / c_size, sum(sensitive_test[test_c_idx] == np.argmax(yeom_pred_2[test_c_idx] * prior_prob, axis=1)) / c_size))
            print('Yeom1_IP:\t%.2f\t%.2f' % (sum(sensitive_test[train_c_idx] == np.argmax(yeom_pred[train_c_idx] * imp_conf[train_c_idx], axis=1)) / c_size, sum(sensitive_test[test_c_idx] == np.argmax(yeom_pred[test_c_idx] * imp_conf[test_c_idx], axis=1)) / c_size))
            print('Yeom2_IP:\t%.2f\t%.2f' % (sum(sensitive_test[train_c_idx] == np.argmax(yeom_pred_2[train_c_idx] * imp_conf[train_c_idx], axis=1)) / c_size, sum(sensitive_test[test_c_idx] == np.argmax(yeom_pred_2[test_c_idx] * imp_conf[test_c_idx], axis=1)) / c_size))
            print('BB:\t%.2f\t%.2f' % (sum(sensitive_test[train_c_idx] == np.argmax(model_conf[train_c_idx], axis=1)) / c_size, sum(sensitive_test[test_c_idx] == np.argmax(model_conf[test_c_idx], axis=1)) / c_size))
            print('CSMIA:\t%.2f\t%.2f' % (sum(sensitive_test[train_c_idx] == csmia_pred[train_c_idx]) / c_size, sum(sensitive_test[test_c_idx] == csmia_pred[test_c_idx]) / c_size))
            print('BB.IP:\t%.2f\t%.2f' % (sum(sensitive_test[train_c_idx] == np.argmax(imp_conf[train_c_idx] * model_conf[train_c_idx], axis=1)) / c_size, sum(sensitive_test[test_c_idx] == np.argmax(imp_conf[test_c_idx] * model_conf[test_c_idx], axis=1)) / c_size))
            
            if threat_level == 'high' and sample_size == 50000:
                whitebox_info, plot_info_dict = get_whitebox_info(target_classifier, true_x, list(range(len(train_x))), target_attr, labels, attribute_dict, max_attr_vals, col_flags, sensitive_test, c_idx=train_c_idx, skip_corr=args.skip_corr)
            else:
                whitebox_info, plot_info_dict = get_whitebox_info(target_classifier, true_x, adv_known_idx, target_attr, labels, attribute_dict, max_attr_vals, col_flags, sensitive_test, skip_corr=args.skip_corr)
                
            if threat_level == 'low':
                idx = train_c_idx + test_c_idx + sk_test_idx
            elif threat_level == 'low2':
                idx = train_c_idx + test_c_idx + sk2_test_idx
            else:
                idx = train_c_idx + test_c_idx + h_test_idx
            
            if args.target_privacy == 'no_privacy':
                pickle.dump([sensitive_test[idx], csmia_pred[idx], imp_conf[idx], model_conf[idx], [wb[idx] for wb in whitebox_info], plot_info_dict, model_aux, imp_aux], open(RESULT_PATH + args.train_dataset + '/' + MODEL + 'no_privacy_' + str(args.attribute) + '_' + threat_level + '_' + str(sample_size) + '_' + str(args.banished_records) + '_' + str(args.run) + '.p', 'wb'))
            else:
                pickle.dump([sensitive_test[idx], csmia_pred[idx], imp_conf[idx], model_conf[idx], [wb[idx] for wb in whitebox_info], plot_info_dict, model_aux, imp_aux], open(RESULT_PATH + args.train_dataset + '/' + MODEL + args.target_privacy + '_' + args.target_dp + '_' + str(args.target_epsilon) + '_' + str(args.attribute) + '_' + threat_level + '_' + str(sample_size) + '_' + str(args.banished_records) + '_' + str(args.run) + '.p', 'wb'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dataset', type=str)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--use_cpu', type=int, default=0)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--save_data', type=int, default=0, help='0: no save data, instead run experiments (default), 1: save data')
    parser.add_argument('--attribute', type=int, default=0, help='senstive attribute to use: 0, 1 or 2')
    parser.add_argument('--candidate_size', type=int, default=int(1e4), help='candidate set size')
    parser.add_argument('--skew_attribute', type=int, default=0, help='Attribute on which to skew the non-iid data sampling 0 (population, default), 1 or 2 -- for Census 1: Income and 2: Race, and for Texas 1: Charges and 2: Ethnicity')
    parser.add_argument('--skew_outcome', type=int, default=0, help='In case skew_attribute = 2, which outcome to skew the distribution upon -- For Census Race: 0 (White, default), 1 (Black) or 3 (Asian), and for Texas Ethnicity: 0 (Hispanic, default) or 1 (Not Hispanic)')
    parser.add_argument('--sensitive_outcome', type=int, default=0, help='In case skew_attribute = 2, this indicates the sensitive outcome -- For Census Race: 0 (White, default), 1 (Black) or 3 (Asian), and for Texas Ethnicity: 0 (Hispanic, default) or 1 (Not Hispanic)')
    parser.add_argument('--banished_records', type=int, default=0, help='if the set of records in banished.p file are to be removed from model training (default:0 no records are removed)')
    parser.add_argument('--skip_corr', type=int, default=0, help='For Texas-100X, whether to skip Race (or Ethnicity) when the target sensitive attribute is Ethnicity (or Race) -- default is not to skip (0)')
    # target and shadow model configuration
    parser.add_argument('--n_shadow', type=int, default=5)
    parser.add_argument('--target_data_size', type=int, default=int(5e4))
    parser.add_argument('--target_test_train_ratio', type=float, default=0.5)
    parser.add_argument('--target_model', type=str, default='nn')
    parser.add_argument('--target_learning_rate', type=float, default=0.01)
    parser.add_argument('--target_batch_size', type=int, default=200)
    parser.add_argument('--target_n_hidden', type=int, default=256)
    parser.add_argument('--target_epochs', type=int, default=100)
    parser.add_argument('--target_l2_ratio', type=float, default=1e-8)
    parser.add_argument('--target_clipping_threshold', type=float, default=1.0)
    parser.add_argument('--target_privacy', type=str, default='no_privacy')
    parser.add_argument('--target_dp', type=str, default='gdp')
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
    
    # Flag to disable GPU
    if args.use_cpu:
    	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

    if args.save_data == 1:
        sample_noniid_data(args)
    else:
        run_experiment(args)
