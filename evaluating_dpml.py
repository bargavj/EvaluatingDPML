from attack import save_data, load_data, train_target_model, yeom_membership_inference, shokri_membership_inference, yeom_attribute_inference
from utilities import log_loss, get_random_features
from sklearn.metrics import roc_curve
import numpy as np
import argparse
import os
import pickle

RESULT_PATH = 'results/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

def run_experiment(args):
    print('-' * 10 + 'TRAIN TARGET' + '-' * 10 + '\n')
    dataset = load_data('target_data.npz', args)
    train_x, train_y, test_x, test_y = dataset
    true_x = np.vstack((train_x, test_x))
    true_y = np.append(train_y, test_y)
    batch_size = args.target_batch_size

    pred_y, membership, test_classes, classifier, aux = train_target_model(
        args=args,
        dataset=dataset,
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
    train_loss, train_acc, test_loss, test_acc = aux
    per_instance_loss = np.array(log_loss(true_y, pred_y))
   
    features = get_random_features(true_x, range(true_x.shape[1]), 5)
    print(features)

    # Yeom's membership inference attack when only train_loss is known 
    pred_membership = yeom_membership_inference(per_instance_loss, membership, train_loss)
    fpr, tpr, thresholds = roc_curve(membership, pred_membership, pos_label=1)
    yeom_mem_adv = tpr[1] - fpr[1]

    # Shokri's membership inference attack based on shadow model training
    shokri_mi_outputs = shokri_membership_inference(args, pred_y, membership, test_classes)
    shokri_mem_adv, _, shokri_mem_confidence, _, _, _, _ = shokri_mi_outputs

    # Yeom's attribute inference attack when train_loss is known - Adversary 4 of Yeom et al.
    pred_membership_all = yeom_attribute_inference(true_x, true_y, classifier, membership, features, train_loss)
    yeom_attr_adv = []
    for pred_membership in pred_membership_all:
        fpr, tpr, thresholds = roc_curve(membership, pred_membership, pos_label=1)
        yeom_attr_adv.append(tpr[1] - fpr[1])
    
    if not os.path.exists(RESULT_PATH+args.train_dataset):
        os.makedirs(RESULT_PATH+args.train_dataset)
    
    if args.target_privacy == 'no_privacy':
        pickle.dump([train_acc, test_acc, train_loss, membership, shokri_mem_adv, shokri_mem_confidence, yeom_mem_adv, per_instance_loss, yeom_attr_adv, pred_membership_all, features], open(RESULT_PATH+args.train_dataset+'/'+args.target_model+'_'+'no_privacy_'+str(args.l2_ratio)+'.p', 'wb'))
    else:
        pickle.dump([train_acc, test_acc, train_loss, membership, shokri_mem_adv, shokri_mem_confidence, yeom_mem_adv, per_instance_loss, yeom_attr_adv, pred_membership_all, features], open(RESULT_PATH+args.train_dataset+'/'+args.target_model+'_'+args.target_privacy+'_'+args.target_dp+'_'+str(args.target_epsilon)+'_'+str(args.run)+'.p', 'wb'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('train_dataset', type=str)
    parser.add_argument('--run', type=int, default=1)
    parser.add_argument('--use_cpu', type=int, default=0)
    parser.add_argument('--save_model', type=int, default=0)
    parser.add_argument('--save_data', type=int, default=0)
    # target and shadow model configuration
    parser.add_argument('--n_shadow', type=int, default=5)
    parser.add_argument('--target_data_size', type=int, default=int(1e4))
    parser.add_argument('--target_test_train_ratio', type=int, default=1)
    parser.add_argument('--target_model', type=str, default='nn')
    parser.add_argument('--target_learning_rate', type=float, default=0.01)
    parser.add_argument('--target_batch_size', type=int, default=200)
    parser.add_argument('--target_n_hidden', type=int, default=256)
    parser.add_argument('--target_epochs', type=int, default=100)
    parser.add_argument('--target_l2_ratio', type=float, default=1e-8)
    parser.add_argument('--target_clipping_threshold', type=float, default=1)
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
    
    # Flag to disable GPU
    if args.use_cpu:
    	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    if args.save_data:
        save_data(args)
    else:
        run_experiment(args)
