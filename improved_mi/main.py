import sys
import os
import argparse
import pickle
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from core.attack import save_data
from core.attack import load_data
from core.attack import train_target_model
from core.attack import yeom_membership_inference
from core.attack import shokri_membership_inference
from core.attack import proposed_membership_inference
from core.attack import evaluate_proposed_membership_inference
from core.utilities import log_loss

RESULT_PATH = 'results/'

if not os.path.exists(RESULT_PATH):
    os.makedirs(RESULT_PATH)

def run_experiment(args):
    print('-' * 10 + 'TRAIN TARGET' + '-' * 10 + '\n')
    dataset = load_data('target_data.npz', args)
    v_dataset = load_data('shadow0_data.npz', args)
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
   
    # Yeom's membership inference attack when only train_loss is known 
    yeom_mi_outputs_1 = yeom_membership_inference(per_instance_loss, membership, train_loss)
    # Yeom's membership inference attack when both train_loss and test_loss are known - Adversary 2 of Yeom et al.
    yeom_mi_outputs_2 = yeom_membership_inference(per_instance_loss, membership, train_loss, test_loss)

    # Shokri's membership inference attack
    shokri_mi_outputs = shokri_membership_inference(args, pred_y, membership, test_classes)

    # Proposed membership inference attacks
    proposed_mi_outputs = proposed_membership_inference(v_dataset, true_x, true_y, classifier, per_instance_loss, args)
    evaluate_proposed_membership_inference(per_instance_loss, membership, proposed_mi_outputs, fpr_threshold=0.01)
    evaluate_proposed_membership_inference(per_instance_loss, membership, proposed_mi_outputs, fpr_threshold=0.01, per_class_thresh=True)

    if not os.path.exists(RESULT_PATH+args.train_dataset):
        os.makedirs(RESULT_PATH+args.train_dataset)
    
    if args.target_privacy == 'no_privacy':
        pickle.dump([aux, membership, per_instance_loss, yeom_mi_outputs_1, yeom_mi_outputs_2, shokri_mi_outputs, proposed_mi_outputs], open(RESULT_PATH+args.train_dataset+'/'+str(args.target_test_train_ratio)+'_'+args.target_model+'_no_privacy_'+str(args.target_l2_ratio)+'_'+str(args.run)+'.p', 'wb'))	
    else:
        pickle.dump([aux, membership, per_instance_loss, yeom_mi_outputs_1, yeom_mi_outputs_2, shokri_mi_outputs, proposed_mi_outputs], open(RESULT_PATH+args.train_dataset+'/'+str(args.target_test_train_ratio)+'_'+args.target_model+'_'+args.target_privacy+'_'+args.target_dp+'_'+str(args.target_epsilon)+'_'+str(args.run)+'.p', 'wb'))

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
    parser.add_argument('--target_test_train_ratio', type=float, default=1)
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
    # Merlin's noise parameters
    parser.add_argument('--attack_noise_type', type=str, default='gaussian')
    parser.add_argument('--attack_noise_coverage', type=str, default='full')
    parser.add_argument('--attack_noise_magnitude', type=float, default=0.01)

    # parse configuration
    args = parser.parse_args()
    print(vars(args))
    
    # Flag to disable GPU
    if args.use_cpu:
    	os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    else:
        os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'

    if args.save_data:
        save_data(args)
    else:
        run_experiment(args)