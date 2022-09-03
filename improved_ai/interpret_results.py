import sys
import os
import argparse
import pickle
import itertools
import random
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))

from core.data_util import get_sensitive_features
from core.utilities import plot_layer_outputs
from core.utilities import get_inference_threshold
from core.utilities import get_ppvs
from core.utilities import fit_model
from core.utilities import make_line_plot
from core.attack import yeom_membership_inference
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from collections import Counter

frame = plt.rcParams['figure.figsize']
new_rc_params = {
	'figure.figsize': [6,5], # [6,4] for main figures and [6,5] for appendix figures
    'figure.autolayout': True,
	'font.size': 20, # 16 for main figures and 20 for appendix figures
	#'text.usetex': True,
	'font.family': 'serif',
    'font.serif': 'Times New Roman',
	'mathtext.fontset': 'stix',
	'xtick.major.pad': '8'
}
plt.rcParams.update(new_rc_params)
fsize = 18 # 16 for main figures and 18 for appendix figures

RESULT_PATH = 'results/'
T = 5
RUNS = range(T)
THREAT_MODEL = ['low', 'low2', 'med', 'high']
SAMPLE_SIZE = [50, 500, 5000, 50000]

safe_colors = {1: '#029386', 2: '#FC5A50', 3: '#DC143C', 4: '#800000', 5: '#FF1493', 6: '#9400D3', 7: '#4B0082', 8: '#222222'}


def get_result(args):
    MODEL = str(args.skew_attribute) + '_' + str(args.skew_outcome) + '_' + str(args.gamma) + '_' + str(args.model) + '_'
    result = {
        'no_privacy': {
            knw: {
                s_size: {
                    run: pickle.load(open(RESULT_PATH + args.dataset + '/' + MODEL + 'no_privacy_' + str(args.attribute) + '_' + knw + '_' + str(s_size) + '_0_' + str(run) + '.p', 'rb')) for run in RUNS
                } for s_size in SAMPLE_SIZE
            } for knw in THREAT_MODEL
        }
    }
    if args.eps != None:
        result[args.eps] = {
            knw: {
                s_size: {
                    run: pickle.load(open(RESULT_PATH + args.dataset + '/' + MODEL + 'grad_pert_' + args.dp + '_' + str(args.eps) + '_' + str(args.attribute) + '_' + knw + '_' + str(s_size) + '_0_' + str(run) + '.p', 'rb')) for run in RUNS
                } for s_size in SAMPLE_SIZE
            } for knw in THREAT_MODEL
        }
    if args.banished_records == 1:
        result['banished'] = {
            knw: {
                s_size: {
                    run: pickle.load(open(RESULT_PATH + args.dataset + '/' + MODEL + 'no_privacy_' + str(args.attribute) + '_' + knw + '_' + str(s_size) + '_1_' + str(run) + '.p', 'rb')) for run in RUNS
                } for s_size in SAMPLE_SIZE
            } for knw in THREAT_MODEL
        }
    return result


def plot_ppvs(args, gt, wb, mc, ip, clfs, plot_cond, data_type=0):
    """
    args: runtime arguments obtained via main function
    gt: ground truth value for sensitive attribute
    wb: white-box attack prediction vector -- this scaled output of neurons
    mc: black-box attack prediction vector -- prediction confidence of model
    ip: imputation model confidence in predicting sensitive attribute
    clfs: meta-classifier that combines wb and mc
    plot_cond: flags that describe how and what lines to plot
    data_type: 0 (default) -- train data, 1 -- test data, 2: hold-out data
    """    
    clfs_wb, clfs_bb = clfs
    ppv_ip = np.array([get_ppvs(gt[run][data_type], ip[run][data_type]) for run in RUNS])
    ppv_wb = np.array([get_ppvs(gt[run][data_type], wb[run][data_type]) for run in RUNS])
    ppv_wb_c0 = np.array([get_ppvs(gt[run][data_type], ip[run][data_type]*wb[run][data_type]) for run in RUNS])
    if args.comb_flag == 1:
        ppv_wb_c1 = np.array([get_ppvs(gt[run][data_type], clfs_wb[run].predict_proba(np.vstack((ip[run][data_type], wb[run][data_type], ip[run][data_type] * wb[run][data_type])).T)[:, 1]) for run in RUNS])
        print('k in WB scatter plot: %d' % sum(clfs_wb[data_type].predict(np.vstack((ip[0][data_type], wb[0][data_type], ip[0][data_type] * wb[0][data_type])).T)))
    else:
        ppv_wb_c1 = np.array([get_ppvs(gt[run][data_type], clfs_wb[run].predict_proba(np.vstack((ip[run][data_type], wb[run][data_type])).T)[:, 1]) for run in RUNS])
        print('k in WB scatter plot: %d' % sum(clfs_wb[data_type].predict(np.vstack((ip[0][data_type], wb[0][data_type])).T)))
    
    ppv_bb = np.array([get_ppvs(gt[run][data_type], mc[run][data_type]) for run in RUNS])
    ppv_bb_c0 = np.array([get_ppvs(gt[run][data_type], ip[run][data_type]*mc[run][data_type]) for run in RUNS])
    if args.comb_flag == 1:
        ppv_bb_c1 = np.array([get_ppvs(gt[run][data_type], clfs_bb[run].predict_proba(np.vstack((ip[run][data_type], mc[run][data_type], ip[run][data_type] * mc[run][data_type])).T)[:, 1]) for run in RUNS])
        print('k in BB scatter plot: %d' % sum(clfs_bb[data_type].predict(np.vstack((ip[0][data_type], mc[0][data_type], ip[0][data_type] * mc[0][data_type])).T)))
    else:
        ppv_bb_c1 = np.array([get_ppvs(gt[run][data_type], clfs_bb[run].predict_proba(np.vstack((ip[run][data_type], mc[run][data_type])).T)[:, 1]) for run in RUNS])
        print('k in BB scatter plot: %d' % sum(clfs_bb[data_type].predict(np.vstack((ip[0][data_type], mc[0][data_type])).T)))
    
    print('Random\t%.2f +/- %.2f' % (np.mean([sum(gt[run][data_type]) for run in RUNS]), np.std([sum(gt[run][data_type]) for run in RUNS])))
    print('\t Top-10 \t Top-50 \t Top-100')
    print('IP\t%.2f +/- %.2f\t%.2f +/- %.2f\t%.2f +/- %.2f' % (np.mean(ppv_ip[:, 9]), np.std(ppv_ip[:, 9]), np.mean(ppv_ip[:, 49]), np.std(ppv_ip[:, 49]), np.mean(ppv_ip[:, 99]), np.std(ppv_ip[:, 99])))
    print('BB\t%.2f +/- %.2f\t%.2f +/- %.2f\t%.2f +/- %.2f' % (np.mean(ppv_bb[:, 9]), np.std(ppv_bb[:, 9]), np.mean(ppv_bb[:, 49]), np.std(ppv_bb[:, 49]), np.mean(ppv_bb[:, 99]), np.std(ppv_bb[:, 99])))
    print('BB.IP\t%.2f +/- %.2f\t%.2f +/- %.2f\t%.2f +/- %.2f' % (np.mean(ppv_bb_c0[:, 9]), np.std(ppv_bb_c0[:, 9]), np.mean(ppv_bb_c0[:, 49]), np.std(ppv_bb_c0[:, 49]), np.mean(ppv_bb_c0[:, 99]), np.std(ppv_bb_c0[:, 99])))
    print('BBXIP\t%.2f +/- %.2f\t%.2f +/- %.2f\t%.2f +/- %.2f' % (np.mean(ppv_bb_c1[:, 9]), np.std(ppv_bb_c1[:, 9]), np.mean(ppv_bb_c1[:, 49]), np.std(ppv_bb_c1[:, 49]), np.mean(ppv_bb_c1[:, 99]), np.std(ppv_bb_c1[:, 99])))
    print('WB\t%.2f +/- %.2f\t%.2f +/- %.2f\t%.2f +/- %.2f' % (np.mean(ppv_wb[:, 9]), np.std(ppv_wb[:, 9]), np.mean(ppv_wb[:, 49]), np.std(ppv_wb[:, 49]), np.mean(ppv_wb[:, 99]), np.std(ppv_wb[:, 99])))
    print('WB.IP\t%.2f +/- %.2f\t%.2f +/- %.2f\t%.2f +/- %.2f' % (np.mean(ppv_wb_c0[:, 9]), np.std(ppv_wb_c0[:, 9]), np.mean(ppv_wb_c0[:, 49]), np.std(ppv_wb_c0[:, 49]), np.mean(ppv_wb_c0[:, 99]), np.std(ppv_wb_c0[:, 99])))
    print('WBXIP\t%.2f +/- %.2f\t%.2f +/- %.2f\t%.2f +/- %.2f' % (np.mean(ppv_wb_c1[:, 9]), np.std(ppv_wb_c1[:, 9]), np.mean(ppv_wb_c1[:, 49]), np.std(ppv_wb_c1[:, 49]), np.mean(ppv_wb_c1[:, 99]), np.std(ppv_wb_c1[:, 99])))
    
    X = list(range(1, args.candidate_size + 1))
    if plot_cond['ip']['flag']:
        make_line_plot(X, ppv_ip, '-', safe_colors[1], 'IP', plot_cond['ip']['xoffset'], plot_cond['ip']['yoffset'], fsize=fsize)
    if plot_cond['wb']['flag']:
        make_line_plot(X, ppv_wb, '-', safe_colors[2], 'WB', plot_cond['wb']['xoffset'], plot_cond['wb']['yoffset'], fsize=fsize)
    if plot_cond['wb.ip']['flag']:
        make_line_plot(X, ppv_wb_c0, '-', safe_colors[3], r'WB$\cdot$IP', plot_cond['wb.ip']['xoffset'], plot_cond['wb.ip']['yoffset'], fsize=fsize)
    if plot_cond['wbXip']['flag']:
        make_line_plot(X, ppv_wb_c1, '-', safe_colors[4], r'WB$\diamondsuit$IP', plot_cond['wbXip']['xoffset'], plot_cond['wbXip']['yoffset'], fsize=fsize)
    if plot_cond['bb']['flag']:
        make_line_plot(X, ppv_bb, '-', safe_colors[5], 'BB', plot_cond['bb']['xoffset'], plot_cond['bb']['yoffset'], fsize=fsize)
    if plot_cond['bb.ip']['flag']:
        make_line_plot(X, ppv_bb_c0, '-', safe_colors[6], r'BB$\cdot$IP', plot_cond['bb.ip']['xoffset'], plot_cond['bb.ip']['yoffset'], fsize=fsize)
    if plot_cond['bbXip']['flag']:
        make_line_plot(X, ppv_bb_c1, '-', safe_colors[7], r'BB$\diamondsuit$IP', plot_cond['bbXip']['xoffset'], plot_cond['bbXip']['yoffset'], fsize=fsize)
    if plot_cond['baseline']['flag']:
        make_line_plot(X, np.array([np.ones(len(X)) * sum(gt[run][data_type]) / len(X) for run in RUNS]), '-.', safe_colors[8], 'Random Guess', plot_cond['baseline']['xoffset'], plot_cond['baseline']['yoffset'], fsize=fsize)
    plt.xlabel('Top-k records')
    plt.ylabel('PPV')
    plt.xscale('log')
    plt.xlim(1, args.candidate_size)
    plt.xticks([1, 1e1, 1e2, 1e3, 1e4])
    plt.yticks([0, 0.2, 0.4, 0.6, 0.8, 1])
    plt.ylim(0, 1)
    plt.tight_layout()
    plt.show()

    
def plot_ppv_change(args, gt, wb, mc, ip, clfs, plot_cond):
    wb_old = {run: {idx: [] for idx in range(3)} for run in RUNS}
    for run in RUNS:
        sensitive_test, _, _, whitebox_info, _, _, _ =  result['no_privacy']['low'][run]
        whitebox_info_k_1, whitebox_info_k_2, whitebox_info_k_5, whitebox_info_k_10 = whitebox_info
        for idx, args.candidate_indices in zip([0, 1, 2], [range(args.candidate_size), range(args.candidate_size, 2*args.candidate_size), range(2*args.candidate_size, len(sensitive_test))]):
            wb_old[run][idx] = whitebox_info_k_10[candidate_indices, args.sensitive_outcome]
    
    clfs_wb_old = [fit_model(args, gt[run][2], ip[run][2], wb_old[run][2]) for run in RUNS]
    ppv_wb1_old = np.array([get_ppvs(gt[run][0], wb_old[run][0]) for run in RUNS])
    ppv_wb1_c0_old = np.array([get_ppvs(gt[run][0], ip[run][0]*wb_old[run][0]) for run in RUNS])
    ppv_wb1_c1_old = np.array([get_ppvs(gt[run][0], clfs_wb_old[run].predict_proba(np.vstack((ip[run][0], wb_old[run][0])).T)[:, 1]) for run in RUNS])
    
    clfs_wb, clfs_bb = clfs
    ppv_ip1 = np.array([get_ppvs(gt[run][0], ip[run][0]) for run in RUNS])
    ppv_wb1 = np.array([get_ppvs(gt[run][0], wb[run][0]) for run in RUNS])
    ppv_wb1_c0 = np.array([get_ppvs(gt[run][0], ip[run][0]*wb[run][0]) for run in RUNS])
    ppv_wb1_c1 = np.array([get_ppvs(gt[run][0], clfs_wb[run].predict_proba(np.vstack((ip[run][0], wb[run][0])).T)[:, 1]) for run in RUNS])
            
    X = list(range(1, args.candidate_size + 1))
    make_line_plot(X, ppv_wb1 - ppv_wb1_old, '-', safe_colors[2], 'Whitebox Attack (WB)', plot_cond['wb']['xoffset'], plot_cond['wb']['yoffset'], fsize=fsize)
    make_line_plot(X, ppv_wb1_c0 - ppv_wb1_c0_old, '-', safe_colors[3], r'WB$\cdot$IP', plot_cond['wb.ip']['xoffset'], plot_cond['wb.ip']['yoffset'], fsize=fsize)
    make_line_plot(X, ppv_wb1_c1 - ppv_wb1_c1_old, '-', safe_colors[4], r'WB$\diamondsuit$IP', plot_cond['wbXip']['xoffset'], plot_cond['wbXip']['yoffset'], fsize=fsize)
    make_line_plot(X, np.array([np.zeros(len(X)) for run in RUNS]), '-.', safe_colors[8], 'No Change', plot_cond['baseline']['xoffset'], plot_cond['baseline']['yoffset'], fsize=fsize)
    plt.xlabel('Top-k records')
    plt.ylabel('Change in PPV')
    plt.xscale('log')
    plt.xlim(1, args.candidate_size)
    plt.xticks([1, 1e1, 1e2, 1e3, 1e4])
    plt.yticks([-1, -0.5, 0, 0.5, 1])
    plt.ylim(-1, 1)
    plt.tight_layout()
    plt.show()


def plot_roc(args, gt, wb, mc, ip):
    mean_fpr = np.linspace(0, 1, args.candidate_size)
    tpr_ip1, tpr_ip2, tpr_wb1, tpr_wb2, tpr_bb1, tpr_bb2 = [], [], [], [], [], []
    for run in RUNS:
        fpr, tpr, _ = roc_curve(gt[run][0], ip[run][0], pos_label=1)
        tpr_ip1.append(np.interp(mean_fpr, fpr, tpr))
        fpr, tpr, _ = roc_curve(gt[run][1], ip[run][1], pos_label=1)
        tpr_ip2.append(np.interp(mean_fpr, fpr, tpr))
        fpr, tpr, _ = roc_curve(gt[run][0], wb[run][0], pos_label=1)
        tpr_wb1.append(np.interp(mean_fpr, fpr, tpr))
        fpr, tpr, _ = roc_curve(gt[run][1], wb[run][1], pos_label=1)
        tpr_wb2.append(np.interp(mean_fpr, fpr, tpr))
        fpr, tpr, _ = roc_curve(gt[run][0], mc[run][0], pos_label=1)
        tpr_bb1.append(np.interp(mean_fpr, fpr, tpr))
        fpr, tpr, _ = roc_curve(gt[run][1], mc[run][1], pos_label=1)
        tpr_bb2.append(np.interp(mean_fpr, fpr, tpr))
    
    make_line_plot(mean_fpr, tpr_ip1, '-', safe_colors[1], 'Imputation on Train', fsize=fsize)
    make_line_plot(mean_fpr, tpr_ip2, '--', safe_colors[1], 'Imputation on Test', fsize=fsize)
    
    make_line_plot(mean_fpr, tpr_wb1, '-', safe_colors[2], 'WhiteBox AI on Train', fsize=fsize)
    make_line_plot(mean_fpr, tpr_wb2, '--', safe_colors[2], 'WhiteBox AI on Test', fsize=fsize)
    
    make_line_plot(mean_fpr, tpr_bb1, '-', safe_colors[5], 'BlackBox AI on Train', fsize=fsize)
    make_line_plot(mean_fpr, tpr_bb2, '--', safe_colors[5], 'BlackBox AI on Test', fsize=fsize)
    
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlim(1 / args.candidate_size, 1)
    plt.ylim(1 / args.candidate_size, 1)
    plt.legend()
    plt.tight_layout()
    plt.show()


def scatter_plot(args, gt, wb, mc, ip, clfs, it):
    N = 5
    clfs_wb, clfs_bb = clfs
    plt.rcParams['figure.figsize'] = [N*3.50, 3.50]
    xx, yy = np.meshgrid(np.arange(0, 1.01, 0.02), np.arange(0, 1.01, 0.02))
    
    if args.dataset == 'census':
        ip_low, ip_high, wb_low = 0.28, 0.5, 0.5
    else:
        ip_low, ip_high, wb_low = 0.0, 0.3, 0.9
    
    vul_idxs = []
    tpls = np.zeros((N, 3))
    f, ax = plt.subplots(1, N, sharey=True, squeeze=False)
    ax[0][0].set_ylabel('Imputation Confidence')
    for i in range(N):
        vul_idx = list(filter(lambda idx: (ip[i][it][idx] > ip_low) and (ip[i][it][idx] < ip_high) and (wb[i][it][idx] > wb_low), range(len(gt[i][it]))))
        vul_idxs.append(list(filter(lambda idx: gt[i][it][idx], vul_idx)))
        tpls[i] = [sum(gt[i][it][vul_idx]), len(vul_idx), 0 if len(vul_idx) == 0 else sum(gt[i][it][vul_idx])/len(vul_idx)]
        print('%d / %d, %.2f PPV' % tuple(tpls[i]))
        
        c_vec = ['#DAA520' if gt_ == 0 else '#DC143C' for gt_ in gt[i][it]]
        ax[0][i].scatter(wb[i][it], ip[i][it], s=3*np.pi, c=c_vec, alpha=0.6)
        ax[0][i].set_xlabel('Whitebox Output')
        Z = clfs_wb[i].predict(np.c_[yy.ravel(), xx.ravel()])
        Z = Z.reshape(xx.shape)
        ax[0][i].contour(xx, yy, Z, levels=0, colors=['black'])
        print(sum(clfs_wb[i].predict(np.vstack((ip[i][it], wb[i][it])).T)), sum(clfs_bb[i].predict(np.vstack((ip[i][it], mc[i][it])).T)))
    print('Mean: %.2f / %.2f, %.2f PPV' % tuple(np.mean(tpls, axis=0)))
    print('Std : %.2f / %.2f, %.2f PPV' % tuple(np.std(tpls, axis=0)))
    
    if args.banished_records == 0:
        for i in range(N):
            ax[0][i].scatter(wb[i][it][vul_idxs[i]], ip[i][it][vul_idxs[i]], s=3*np.pi, c='blue', alpha=0.6)
        if args.sample_size == 5000:
            pickle.dump(vul_idxs, open('data/' + args.dataset + '/' + str(args.attribute) + '_' + args.adv_knowledge + '_' + 'banished_records.p', 'wb'))
    else:
        banished_idx = pickle.load(open('data/' + args.dataset + '/' + str(args.attribute) + '_' + args.adv_knowledge + '_' + 'banished_records.p', 'rb'))
        tpls = np.zeros((N, 3))
        for i in range(N):
            tpls[i] = [len(set(banished_idx[i]).intersection(set(vul_idxs[i]))), len(banished_idx[i]), len(set(vul_idxs[i]) - set(banished_idx[i]))]
            print('%d / %d records still vulnerable, %d new vulnerable records' % tuple(tpls[i]))
            ax[0][i].scatter(wb[i][it][banished_idx[i]], ip[i][it][banished_idx[i]], s=3*np.pi, c='blue', alpha=0.6)
        print('Mean: %.2f / %.2f still vulnerable, %.2f new vulnerable' % tuple(np.mean(tpls, axis=0)))
        print('Std : %.2f / %.2f still vulnerable, %.2f new vulnerable' % tuple(np.std(tpls, axis=0)))
    
    f.tight_layout()
    plt.show()
    
    f, ax = plt.subplots(1, N, sharey=True, squeeze=False)
    ax[0][0].set_ylabel('Imputation Confidence')
    for i in range(N):
        c_vec = ['#DAA520' if gt_ == 0 else '#DC143C' for gt_ in gt[i][it]]
        ax[0][i].scatter(mc[i][it], ip[i][it], s=3*np.pi, c=c_vec, alpha=0.6)
        ax[0][i].set_xlabel('Model Confidence')
        Z = clfs_bb[i].predict(np.c_[yy.ravel(), xx.ravel()])
        Z = Z.reshape(xx.shape)
        ax[0][i].contour(xx, yy, Z, levels=0, colors=['black'])
    f.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('--model', type=str, default='nn')
    parser.add_argument('--skew_attribute', type=int, default=0, help='Attribute on which to skew the non-iid data sampling 0 (population, default), 1 or 2 -- for Census 1: Income and 2: Race, and for Texas 1: Charges and 2: Ethnicity')
    parser.add_argument('--skew_outcome', type=int, default=0, help='In case skew_attribute = 1, which outcome the distribution is skewed upon -- For Census Race: 0 (White, default), 1 (Black) or 3 (Asian), and for Texas Ethnicity: 0 (Hispanic, default) or 1 (Not Hispanic)')
    parser.add_argument('--comb_flag', type=int, default=0, help='0 (default): just use 2 featues for combining using meta model, 1: use 3 features for training meta model')
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--dp', type=str, default='gdp')
    parser.add_argument('--eps', type=float, default=None)
    parser.add_argument('--attribute', type=int, default=0)
    parser.add_argument('--sensitive_outcome', type=int, default=1)
    parser.add_argument('--candidate_size', type=int, default=int(1e4))
    parser.add_argument('--adv_knowledge', type=str, default='low')
    parser.add_argument('--sample_size', type=int, default=50000)
    parser.add_argument('--banished_records', type=int, default=0)
    args = parser.parse_args()
    print(vars(args))
    candidate_size = args.candidate_size
    sensitive_outcome = args.sensitive_outcome
        
    result = get_result(args)
    
    gt = {run: {idx: [] for idx in range(3)} for run in RUNS}
    wb = {run: {idx: [] for idx in range(3)} for run in RUNS}
    mc = {run: {idx: [] for idx in range(3)} for run in RUNS}
    ip = {run: {idx: [] for idx in range(3)} for run in RUNS}
    imp_auxs, model_auxs = np.zeros((4, T)), np.zeros((4, T))
    pmc_acc, ip_acc, yeom_acc, csmia_acc, mc_acc, wb_acc, mc_comb_acc, wb_comb_acc = np.zeros((3, T)), np.zeros((3, T)), np.zeros((3, T)), np.zeros((3, T)), np.zeros((3, T)), np.zeros((3, T)), np.zeros((3, T)), np.zeros((3, T))
    corr_vals = np.zeros((100, T))
    
    for run in RUNS:
        v = result['no_privacy'][args.adv_knowledge][args.sample_size][run]
        if args.eps != None:
            v = result[args.eps][args.adv_knowledge][args.sample_size][run]
        elif args.banished_records == 1:
            v = result['banished'][args.adv_knowledge][args.sample_size][run]
        sensitive_test, csmia_pred, _, model_conf, whitebox_info, plot_info_dict, model_aux, _ =  v
        _, _, imp_conf, _, _, _, _, imp_aux = result['no_privacy'][args.adv_knowledge][args.sample_size][run]
        
        model_auxs[:, run] = model_aux
        imp_auxs[:, run] = imp_aux
        if args.dataset == 'purchase_100_binary':
            labels = ['0', '1']
        else:
            target_attrs, attribute_dict, _, _ = get_sensitive_features(args.dataset, np.zeros((1,1)))
            labels = list(attribute_dict[target_attrs[args.attribute]].values())
        whitebox_info_k_1, whitebox_info_k_10, whitebox_info_k_100 = whitebox_info
        
        # selecting which WB attack to evaluate
        whitebox_info_k = whitebox_info_k_10
        corr_vals[:, run] = plot_info_dict[str(sensitive_outcome)]['correlation_vals']
        #if run == 0:
        #    plot_layer_outputs(informative_neurons = plot_info_dict[str(sensitive_outcome)]['informative_neurons'], plot_info = plot_info_dict[str(sensitive_outcome)]['plot_info'], pos_label = labels[int(sensitive_outcome)], neg_label = 'Not '+labels[int(sensitive_outcome)])
        
        train_loss, train_acc, test_loss, test_acc = model_aux
        yeom_pred = yeom_membership_inference(-np.log(model_conf), None, train_loss, test_loss)
        
        cc = Counter(sensitive_test[range(candidate_size)])
        for k, v in cc.items():
            print(attribute_dict[target_attrs[args.attribute]][k], v)
        
        for idx, txt, candidate_indices in zip([0, 1, 2], ['train', 'test', 'holdout'], [range(candidate_size), range(candidate_size, 2*candidate_size), range(2*candidate_size, len(sensitive_test))]):
            pmc_acc[idx][run] = max(Counter(sensitive_test[candidate_indices]).values()) / candidate_size
            
            prior_prob = np.zeros(len(labels))
            for k, v in Counter(sensitive_test[candidate_indices]).items():
                prior_prob[int(k)] = v / candidate_size
            yeom_acc[idx][run] = sum(sensitive_test[candidate_indices] == np.argmax(yeom_pred[candidate_indices] * prior_prob, axis=1)) / candidate_size
            
            csmia_acc[idx][run] = sum(sensitive_test[candidate_indices] == csmia_pred[candidate_indices]) / candidate_size 
            
            ip_acc[idx][run] = sum(sensitive_test[candidate_indices] == np.argmax(imp_conf[candidate_indices], axis=1)) / candidate_size
            
            mc_acc[idx][run] = sum(sensitive_test[candidate_indices] == np.argmax(model_conf[candidate_indices], axis=1)) / candidate_size
            
            mc_comb_acc[idx][run] = sum(sensitive_test[candidate_indices] == np.argmax(imp_conf[candidate_indices] * model_conf[candidate_indices], axis=1)) / candidate_size
            
            wb_acc[idx][run] = sum(sensitive_test[candidate_indices] == np.argmax(whitebox_info_k[candidate_indices], axis=1)) / candidate_size
            
            wb_comb_acc[idx][run] = sum(sensitive_test[candidate_indices] == np.argmax(imp_conf[candidate_indices] * whitebox_info_k[candidate_indices], axis=1)) / candidate_size
            
            gt[run][idx] = np.where(sensitive_test[candidate_indices] == sensitive_outcome, 1, 0)
            wb[run][idx] = whitebox_info_k[candidate_indices, sensitive_outcome]
            mc[run][idx] = model_conf[candidate_indices, sensitive_outcome]
            ip[run][idx] = imp_conf[candidate_indices, sensitive_outcome]
    
    print('Model Performance:\nTrain Loss:\t%.3f +/- %.3f\nTrain Acc:\t%.2f +/- %.2f\nTest Loss:\t%.3f +/- %.3f\nTest Acc:\t%.2f +/- %.2f' % (np.mean(model_auxs[0]), np.std(model_auxs[0]), np.mean(model_auxs[1]), np.std(model_auxs[1]), np.mean(model_auxs[2]), np.std(model_auxs[2]), np.mean(model_auxs[3]), np.std(model_auxs[3])))
    
    print('Imputation Performance:\nTrain Loss:\t%.3f +/- %.3f\nTrain Acc:\t%.2f +/- %.2f\nTest Loss:\t%.3f +/- %.3f\nTest Acc:\t%.2f +/- %.2f' % (np.mean(imp_auxs[0]), np.std(imp_auxs[0]), np.mean(imp_auxs[1]), np.std(imp_auxs[1]), np.mean(imp_auxs[2]), np.std(imp_auxs[2]), np.mean(imp_auxs[3]), np.std(imp_auxs[3])))
    
    print('\t Train AI Acc \t Test AI Acc')
    print('PMC\t%.2f +/- %.2f\t%.2f +/- %.2f' % (np.mean(pmc_acc[0]), np.std(pmc_acc[0]), np.mean(pmc_acc[1]), np.std(pmc_acc[1])))
    print('IP\t%.2f +/- %.2f\t%.2f +/- %.2f' % (np.mean(ip_acc[0]), np.std(ip_acc[0]), np.mean(ip_acc[1]), np.std(ip_acc[1])))
    print('Yeom\t%.2f +/- %.2f\t%.2f +/- %.2f' % (np.mean(yeom_acc[0]), np.std(yeom_acc[0]), np.mean(yeom_acc[1]), np.std(yeom_acc[1])))
    print('CSMIA\t%.2f +/- %.2f\t%.2f +/- %.2f' % (np.mean(csmia_acc[0]), np.std(csmia_acc[0]), np.mean(csmia_acc[1]), np.std(csmia_acc[1])))
    print('BB\t%.2f +/- %.2f\t%.2f +/- %.2f' % (np.mean(mc_acc[0]), np.std(mc_acc[0]), np.mean(mc_acc[1]), np.std(mc_acc[1])))
    print('BB*IP\t%.2f +/- %.2f\t%.2f +/- %.2f' % (np.mean(mc_comb_acc[0]), np.std(mc_comb_acc[0]), np.mean(mc_comb_acc[1]), np.std(mc_comb_acc[1])))
    print('WB\t%.2f +/- %.2f\t%.2f +/- %.2f' % (np.mean(wb_acc[0]), np.std(wb_acc[0]), np.mean(wb_acc[1]), np.std(wb_acc[1])))
    print('WB*IP\t%.2f +/- %.2f\t%.2f +/- %.2f' % (np.mean(wb_comb_acc[0]), np.std(wb_comb_acc[0]), np.mean(wb_comb_acc[1]), np.std(wb_comb_acc[1])))
    
    print('\nTop-10 Neuron Correlation Values:\t%.2f +/- %.2f' % (np.mean(corr_vals[:10]), np.std(corr_vals[:10])))
    for i in range(10):
        print('Neuron #%d:\t%.2f +/- %.2f' % (i, np.mean(corr_vals[i]), np.std(corr_vals[i])))
    
    clfs_wb = [fit_model(args, gt[run][2], ip[run][2], wb[run][2]) for run in RUNS]
    clfs_bb = [fit_model(args, gt[run][2], ip[run][2], mc[run][2]) for run in RUNS]
    
    # change the values in plot_cond to get the desired plots
    plot_cond = {
        'ip': {'flag': True, 'xoffset': 100, 'yoffset': 0.04},
        'wb': {'flag': True, 'xoffset': 2, 'yoffset': 0.02},
        'wb.ip': {'flag': True, 'xoffset': 7, 'yoffset': -0.08},
        'wbXip': {'flag': True, 'xoffset': 100, 'yoffset': 0.04},
        'bb': {'flag': False, 'xoffset': 1, 'yoffset': 0.05},
        'bb.ip': {'flag': False, 'xoffset': 1, 'yoffset': -0.06},
        'bbXip': {'flag': False, 'xoffset': 1000, 'yoffset': 0.05},
        'baseline': {'flag': True, 'xoffset': 5, 'yoffset': 0.05}
    }
    print('*'*5 + ' Train Set ' + '*'*5)
    plot_ppvs(args, gt, wb, mc, ip, (clfs_wb, clfs_bb), plot_cond)
    print('*'*5 + ' Test Set ' + '*'*5)
    plot_ppvs(args, gt, wb, mc, ip, (clfs_wb, clfs_bb), plot_cond, data_type=1)
    '''
    plot_cond = {
        'wb': {'xoffset': 1, 'yoffset': -0.16},
        'wb.ip': {'xoffset': 4, 'yoffset': -0.05},
        'wbXip': {'xoffset': 2, 'yoffset': 0.1},
        'baseline': {'xoffset': 800, 'yoffset': -0.2}
    }
    plot_ppv_change(args, gt, wb, mc, ip, (clfs_wb, clfs_bb), plot_cond)
    '''
    #plot_roc(args, gt, wb, mc, ip)
    scatter_plot(args, gt, wb, mc, ip, (clfs_wb, clfs_bb), 0)
