from sklearn.metrics import roc_curve, confusion_matrix
from scipy import stats
from utilities import get_tp_adv_ppv, get_ppv, get_inference_threshold, plot_histogram, plot_sign_histogram
from attack import evaluate_proposed_membership_inference
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse


EPS = list(np.arange(0.1, 100, 0.01))
EPS2 = list(np.arange(0.1, 100, 0.01))
#EPSILONS = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
EPSILONS = [1.0, 10.0, 100.0]
PERTURBATION = 'grad_pert_'
DP = ['gdp_']
TYPE = ['o-', '.-']
DP_LABELS = ['GDP', 'RDP']
RUNS = range(5)
A, B = len(EPSILONS), len(RUNS)
ALPHAS = np.arange(0.01, 1, 0.01)
delta = 1e-5

plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'
#plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 20})


def f(eps, delta, alpha):
	return max(0, 1 - delta - np.exp(eps) * alpha, np.exp(-eps) * (1 - delta - alpha))


def adv_lim(eps, delta, alpha):
	return 1 - f(eps, delta, alpha) - alpha


def ppv_lim(eps, delta, alpha):
	return (1 - f(eps, delta, alpha)) / (1 - f(eps, delta, alpha) + gamma * alpha)


def improved_limit(epsilons):
	return [max([adv_lim(eps, delta, alpha) for alpha in ALPHAS]) for eps in epsilons]


def yeoms_limit(epsilons):
	return [np.exp(eps) - 1 for eps in epsilons]


def get_data():
	result = {}
	for dp in DP:
		epsilons = {}
		for eps in EPSILONS:
			if eps > 100 and dp == 'gdp_':
				continue
			runs = {}
			for run in RUNS:
				runs[run] = list(pickle.load(open(DATA_PATH+MODEL+PERTURBATION+dp+str(eps)+'_'+str(run+1)+'.p', 'rb')))
			epsilons[eps] = runs
		result[dp] = epsilons
	runs = {}
	for run in RUNS:
		runs[run] = list(pickle.load(open(DATA_PATH+MODEL+'no_privacy_'+str(args.l2_ratio)+'_'+str(run+1)+'.p', 'rb')))
	result['no_privacy'] = runs
	return result


def pretty_position(X, Y, pos):
	return ((X[pos] + X[pos+1]) / 2, (Y[pos] + Y[pos+1]) / 2)


def get_pred_mem(per_instance_loss, proposed_mi_outputs, proposed_ai_outputs=None, i=None, method=1, fpr_threshold=None):
	v_membership, v_per_instance_loss, v_counts, counts = proposed_mi_outputs
	if proposed_ai_outputs == None:
		if method == 1:
			thresh = get_inference_threshold(-v_per_instance_loss, v_membership, fpr_threshold)
			print(thresh)
			return np.where(per_instance_loss <= -thresh, 1, 0)
		else:
			thresh = get_inference_threshold(v_counts, v_membership, fpr_threshold)
			print(thresh)
			return np.where(counts >= thresh, 1, 0)
	else:
		true_attribute_value_all, low_per_instance_loss_all, high_per_instance_loss_all, low_counts_all, high_counts_all = proposed_ai_outputs
		high_prob = np.sum(true_attribute_value_all[i]) / len(true_attribute_value_all[i])
		low_prob = 1 - high_prob
		if method == 1:
			thresh = get_inference_threshold(-v_per_instance_loss, v_membership, fpr_threshold)
			low_mem = np.where(low_per_instance_loss_all[i] <= -thresh, 1, 0)
			high_mem = np.where(high_per_instance_loss_all[i] <= -thresh, 1, 0)
		else:
			thresh = get_inference_threshold(v_counts, v_membership, fpr_threshold)
			low_mem = np.where(low_counts_all[i] >= thresh, 1, 0)
			high_mem = np.where(high_counts_all[i] >= thresh, 1, 0)
		pred_attribute_value = [np.argmax([low_prob * a, high_prob * b]) for a, b in zip(low_mem, high_mem)]
		mask = [a | b for a, b in zip(low_mem, high_mem)]
		return mask & (pred_attribute_value ^ true_attribute_value_all[i] ^ [1]*len(pred_attribute_value))

def plot_distributions(pred_vector, true_vector, method=1):
	fpr, tpr, phi = roc_curve(true_vector, pred_vector, pos_label=1)
	fpr, tpr, phi = np.array(fpr), np.array(tpr), np.array(phi)
	if method == 1:
		fpr = 1 - fpr
		tpr = 1 - tpr
	PPV_A = tpr / (tpr + gamma * fpr)
	Adv_A = tpr - fpr
	fig, ax1 = plt.subplots()
	if method == 1:
		phi, fpr, Adv_A, PPV_A = phi[:-1], fpr[:-1], Adv_A[:-1], PPV_A[:-1]
	ax1.plot(phi, Adv_A, label="Adv", color='black')
	ax1.plot(phi, PPV_A, label="PPV", color='black')
	ax2 = ax1.twinx()
	ax2.plot(phi, fpr, label="FPR", color='black', linestyle='dashed')
	if method == 1:
		ax1.set_xscale('log')
		ax1.annotate('$Adv_\mathcal{A}$', pretty_position(phi, Adv_A, np.argmax(Adv_A)), textcoords="offset points", xytext=(-18,-20), ha='right')
		ax1.annotate('$PPV_\mathcal{A}$', pretty_position(phi, PPV_A, -50), textcoords="offset points", xytext=(-10,10), ha='right')
		ax2.annotate('FPR ($\\alpha$)', pretty_position(phi, fpr, 0), textcoords="offset points", xytext=(-10,-10), ha='right')
	else:
		ax1.annotate('$Adv_\mathcal{A}$', pretty_position(phi, Adv_A, np.argmax(Adv_A)), textcoords="offset points", xytext=(-20,0), ha='right')
		ax1.annotate('$PPV_\mathcal{A}$', pretty_position(phi, PPV_A, 5), textcoords="offset points", xytext=(-10,10), ha='right')
		ax2.annotate('FPR ($\\alpha$)', pretty_position(phi, fpr, -5), textcoords="offset points", xytext=(0,-30), ha='left')
		ax1.set_xticks(np.arange(0, 101, step=20))
	ax1.set_xlabel('Decision Function $\phi$')
	ax1.set_ylabel('Privacy Leakage Metrics')
	ax2.set_ylabel('False Positive Rate')
	ax1.set_yticks(np.arange(0, 1.1, step=0.2))
	ax2.set_yticks(np.arange(0, 1.1, step=0.2))
	fig.tight_layout()
	plt.show()


def generate_plots(result):
	train_accs, baseline_acc = np.zeros(B), np.zeros(B)
	adv_y_mi_1, adv_y_mi_2, adv_y_ai_1, adv_y_ai_2, adv_p_mi_1, adv_p_mi_2, adv_p_ai_1, adv_p_ai_2 = np.zeros(B), np.zeros(B), np.zeros(5*B), np.zeros(5*B), np.zeros(B), np.zeros(B), np.zeros(5*B), np.zeros(5*B)
	ppv_y_mi_1, ppv_y_mi_2, ppv_y_ai_1, ppv_y_ai_2, ppv_p_mi_1, ppv_p_mi_2, ppv_p_ai_1, ppv_p_ai_2 = np.zeros(B), np.zeros(B), np.zeros(5*B), np.zeros(5*B), np.zeros(B), np.zeros(B), np.zeros(5*B), np.zeros(5*B)
	tp_y_mi_1, tp_y_mi_2, tp_y_ai_1, tp_y_ai_2, tp_p_mi_1, tp_p_mi_2, tp_p_ai_1, tp_p_ai_2 = np.zeros(B), np.zeros(B), np.zeros(5*B), np.zeros(5*B), np.zeros(B), np.zeros(B), np.zeros(5*B), np.zeros(5*B)
	#pred1, pred2, pred3, pred4 = [], [], [], []
	for run in RUNS:
		aux, membership, per_instance_loss, features, yeom_mi_outputs_1, yeom_mi_outputs_2, yeom_ai_outputs_1, yeom_ai_outputs_2, proposed_mi_outputs, proposed_ai_outputs = result['no_privacy'][run]
		train_loss, train_acc, test_loss, test_acc = aux
		v_membership, v_per_instance_loss, v_counts, counts = proposed_mi_outputs
		print('Yeom phi: %f' % train_loss)
		plot_histogram(per_instance_loss)
		#plot_distributions(per_instance_loss, membership)
		plot_sign_histogram(membership, counts, 100)
		#plot_distributions(counts, membership, 2)
		baseline_acc[run] = test_acc
		train_accs[run] = train_acc
		
		print("MI 1")
		tp, adv, ppv = get_tp_adv_ppv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, method=1, fpr_threshold=alpha))
		tp_p_mi_1[run], adv_p_mi_1[run], ppv_p_mi_1[run] = tp, adv, ppv
		print("MI 2")
		tp, adv, ppv = get_tp_adv_ppv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, method=2, fpr_threshold=alpha))
		tp_p_mi_2[run], adv_p_mi_2[run], ppv_p_mi_2[run] = tp, adv, ppv
		tp, adv, ppv = get_tp_adv_ppv(membership, yeom_mi_outputs_1)
		tp_y_mi_1[run], adv_y_mi_1[run], ppv_y_mi_1[run] = tp, adv, ppv
		tp, adv, ppv = get_tp_adv_ppv(membership, yeom_mi_outputs_2)
		tp_y_mi_2[run], adv_y_mi_2[run], ppv_y_mi_2[run] = tp, adv, ppv
		'''
		for i in range(5):
			tp, adv, ppv = get_tp_adv_ppv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, proposed_ai_outputs, i, method=1, fpr_threshold=alpha))
			tp_p_ai_1[run*5 + i], adv_p_ai_1[run*5 + i], ppv_p_ai_1[run*5 + i] = tp, adv, ppv
			tp, adv, ppv = get_tp_adv_ppv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, proposed_ai_outputs, i, method=2, fpr_threshold=alpha))
			tp_p_ai_2[run*5 + i], adv_p_ai_2[run*5 + i], ppv_p_ai_2[run*5 + i] = tp, adv, ppv
			tp, adv, ppv = get_tp_adv_ppv(membership, yeom_ai_outputs_1[i])
			tp_y_ai_1[run*5 + i], adv_y_ai_1[run*5 + i], ppv_y_ai_1[run*5 + i] = tp, adv, ppv
			tp, adv, ppv = get_tp_adv_ppv(membership, yeom_ai_outputs_2[i])
			tp_y_ai_2[run*5 + i], adv_y_ai_2[run*5 + i], ppv_y_ai_2[run*5 + i] = tp, adv, ppv
		'''
		#pred1.append(yeom_mi_outputs_1)
		#pred2.append(yeom_mi_outputs_2)
		#pred3.append(get_pred_mem(per_instance_loss, proposed_mi_outputs, method=1, fpr_threshold=alpha))
		#pred4.append(get_pred_mem(per_instance_loss, proposed_mi_outputs, method=2, fpr_threshold=alpha))
	baseline_acc = np.mean(baseline_acc)
	print(np.mean(train_accs), baseline_acc)
	print('Yeom MI 1:\n TP: %d, Adv: %f, PPV: %f' % (np.mean(tp_y_mi_1), np.mean(adv_y_mi_1), np.mean(ppv_y_mi_1)))
	#print('Yeom MI 2:\n TP: %d, Adv: %f, PPV: %f' % (np.mean(tp_y_mi_2), np.mean(adv_y_mi_2), np.mean(ppv_y_mi_2)))
	print('Proposed MI 1:\n TP: %d, Adv: %f, PPV: %f' % (np.mean(tp_p_mi_1), np.mean(adv_p_mi_1), np.mean(ppv_p_mi_1)))
	print('Proposed MI 2:\n TP: %d, Adv: %f, PPV: %f' % (np.mean(tp_p_mi_2), np.mean(adv_p_mi_2), np.mean(ppv_p_mi_2)))
	#print('\nYeom AI 1:\n TP: %d, Adv: %f, PPV: %f' % (np.mean(tp_y_ai_1), np.mean(adv_y_ai_1), np.mean(ppv_y_ai_1)))
	#print('Yeom AI 2:\n TP: %d, Adv: %f, PPV: %f' % (np.mean(tp_y_ai_2), np.mean(adv_y_ai_2), np.mean(ppv_y_ai_2)))
	print('\nProposed AI 1:\n TP: %d, Adv: %f, PPV: %f' % (np.mean(tp_p_ai_1), np.mean(adv_p_ai_1), np.mean(ppv_p_ai_1)))
	print('Proposed AI 2:\n TP: %d, Adv: %f, PPV: %f' % (np.mean(tp_p_ai_2), np.mean(adv_p_ai_2), np.mean(ppv_p_ai_2)))
	'''
	print('MI Results on Non private model')
	print('Yeom method 1')
	print('Adv: %f, PPV: %f' % (np.mean(adv_y_mi_1), np.mean(ppv_y_mi_1)))
	ppv_across_runs(membership, np.sum(np.array(pred1), axis=0))
	print('Yeom method 2')
	print('Adv: %f, PPV: %f' % (np.mean(adv_y_mi_2), np.mean(ppv_y_mi_2)))
	ppv_across_runs(membership, np.sum(np.array(pred2), axis=0))
	print('Our method 1')
	print('Adv: %f, PPV: %f' % (np.mean(adv_p_mi_1), np.mean(ppv_p_mi_1)))
	ppv_across_runs(membership, np.sum(np.array(pred3), axis=0))
	print('Our method 2')
	print('Adv: %f, PPV: %f' % (np.mean(adv_p_mi_2), np.mean(ppv_p_mi_2)))
	ppv_across_runs(membership, np.sum(np.array(pred4), axis=0))
	'''
	color = 0.1
	y = dict()
	for dp in DP:
		test_acc_vec = np.zeros((A, B))
		adv_y_mi_1, adv_y_mi_2, adv_y_ai_1, adv_y_ai_2, adv_p_mi_1, adv_p_mi_2, adv_p_ai_1, adv_p_ai_2 = np.zeros((A, B)), np.zeros((A, B)), np.zeros((A, 5*B)), np.zeros((A, 5*B)), np.zeros((A, B)), np.zeros((A, B)), np.zeros((A, 5*B)), np.zeros((A, 5*B))
		ppv_y_mi_1, ppv_y_mi_2, ppv_y_ai_1, ppv_y_ai_2, ppv_p_mi_1, ppv_p_mi_2, ppv_p_ai_1, ppv_p_ai_2 = np.zeros((A, B)), np.zeros((A, B)), np.zeros((A, 5*B)), np.zeros((A, 5*B)), np.zeros((A, B)), np.zeros((A, B)), np.zeros((A, 5*B)), np.zeros((A, 5*B))
		tp_y_mi_1, tp_y_mi_2, tp_y_ai_1, tp_y_ai_2, tp_p_mi_1, tp_p_mi_2, tp_p_ai_1, tp_p_ai_2 = np.zeros((A, B)), np.zeros((A, B)), np.zeros((A, 5*B)), np.zeros((A, 5*B)), np.zeros((A, B)), np.zeros((A, B)), np.zeros((A, 5*B)), np.zeros((A, 5*B))
		for a, eps in enumerate(EPSILONS):
			#pred1, pred2, pred3, pred4 = [], [], [], []
			for run in RUNS:
				aux, membership, per_instance_loss, features, yeom_mi_outputs_1, yeom_mi_outputs_2, yeom_ai_outputs_1, yeom_ai_outputs_2, proposed_mi_outputs, proposed_ai_outputs = result[dp][eps][run]
				train_loss, train_acc, test_loss, test_acc = aux
				v_membership, v_per_instance_loss, v_counts, counts = proposed_mi_outputs
				test_acc_vec[a, run] = test_acc
				#print(eps, run)
				#plot_histogram(per_instance_loss)
				#plot_distributions(per_instance_loss, membership)
				#plot_sign_histogram(membership, counts, 100)
				#plot_distributions(counts, membership, 2)
				tp, adv, ppv = get_tp_adv_ppv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, method=1, fpr_threshold=alpha))
				tp_p_mi_1[a, run], adv_p_mi_1[a, run], ppv_p_mi_1[a, run] = tp, adv, ppv
				tp, adv, ppv = get_tp_adv_ppv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, method=2, fpr_threshold=alpha))
				tp_p_mi_2[a, run], adv_p_mi_2[a, run], ppv_p_mi_2[a, run] = tp, adv, ppv
				tp, adv, ppv = get_tp_adv_ppv(membership, yeom_mi_outputs_1)
				tp_y_mi_1[a, run], adv_y_mi_1[a, run], ppv_y_mi_1[a, run] = tp, adv, ppv
				tp, adv, ppv = get_tp_adv_ppv(membership, yeom_mi_outputs_2)
				tp_y_mi_2[a, run], adv_y_mi_2[a, run], ppv_y_mi_2[a, run] = tp, adv, ppv
				for i in range(5):
					tp, adv, ppv = get_tp_adv_ppv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, proposed_ai_outputs, i, method=1, fpr_threshold=alpha))
					tp_p_ai_1[a, run*5 + i], adv_p_ai_1[a, run*5 + i], ppv_p_ai_1[a, run*5 + i] = tp, adv, ppv
					tp, adv, ppv = get_tp_adv_ppv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, proposed_ai_outputs, i, method=2, fpr_threshold=alpha))
					tp_p_ai_2[a, run*5 + i], adv_p_ai_2[a, run*5 + i], ppv_p_ai_2[a, run*5 + i] = tp, adv, ppv
					tp, adv, ppv = get_tp_adv_ppv(membership, yeom_ai_outputs_1[i])
					tp_y_ai_1[a, run*5 + i], adv_y_ai_1[a, run*5 + i], ppv_y_ai_1[a, run*5 + i] = tp, adv, ppv
					tp, adv, ppv = get_tp_adv_ppv(membership, yeom_ai_outputs_2[i])
					tp_y_ai_2[a, run*5 + i], adv_y_ai_2[a, run*5 + i], ppv_y_ai_2[a, run*5 + i] = tp, adv, ppv
				#pred1.append(yeom_mi_outputs_1)
				#pred2.append(yeom_mi_outputs_2)
				#pred3.append(get_pred_mem(per_instance_loss, proposed_mi_outputs, method=1, fpr_threshold=alpha))
				#pred4.append(get_pred_mem(per_instance_loss, proposed_mi_outputs, method=2, fpr_threshold=alpha))
			'''
			print('\n'+str(eps)+'\n')
			print('Yeom method 1')
			ppv_across_runs(membership, np.sum(np.array(pred1), axis=0))
			print('Yeom method 2')
			ppv_across_runs(membership, np.sum(np.array(pred2), axis=0))
			print('Our method 1')
			ppv_across_runs(membership, np.sum(np.array(pred3), axis=0))
			print('Our method 2')
			ppv_across_runs(membership, np.sum(np.array(pred4), axis=0))
			'''
			print('\n'+str(eps)+'\n')
			#print('Yeom MI 1:\n TP: %d, Adv: %f, PPV: %f' % (np.mean(tp_y_mi_1[a]), np.mean(adv_y_mi_1[a]), np.mean(ppv_y_mi_1[a])))
			#print('Yeom MI 2:\n TP: %d, Adv: %f, PPV: %f' % (np.mean(tp_y_mi_2[a]), np.mean(adv_y_mi_2[a]), np.mean(ppv_y_mi_2[a])))
			print('Proposed MI 1:\n TP: %d, Adv: %f, PPV: %f' % (np.mean(tp_p_mi_1[a]), np.mean(adv_p_mi_1[a]), np.mean(ppv_p_mi_1[a])))
			print('Proposed MI 2:\n TP: %d, Adv: %f, PPV: %f' % (np.mean(tp_p_mi_2[a]), np.mean(adv_p_mi_2[a]), np.mean(ppv_p_mi_2[a])))
			#print('\nYeom AI 1:\n TP: %d, Adv: %f, PPV: %f' % (np.mean(tp_y_ai_1[a]), np.mean(adv_y_ai_1[a]), np.mean(ppv_y_ai_1[a])))
			#print('Yeom AI 2:\n TP: %d, Adv: %f, PPV: %f' % (np.mean(tp_y_ai_2[a]), np.mean(adv_y_ai_2[a]), np.mean(ppv_y_ai_2[a])))
			print('\nProposed AI 1:\n TP: %d, Adv: %f, PPV: %f' % (np.mean(tp_p_ai_1[a]), np.mean(adv_p_ai_1[a]), np.mean(ppv_p_ai_1[a])))
			print('Proposed AI 2:\n TP: %d, Adv: %f, PPV: %f' % (np.mean(tp_p_ai_2[a]), np.mean(adv_p_ai_2[a]), np.mean(ppv_p_ai_2[a])))
		if args.plot == 'acc':
			y[dp] = 1 - np.mean(test_acc_vec, axis=1) / baseline_acc
			plt.errorbar(EPSILONS, 1 - np.mean(test_acc_vec, axis=1) / baseline_acc, yerr=np.std(test_acc_vec, axis=1), color=str(color), fmt='.-', capsize=2, label=DP_LABELS[DP.index(dp)])
		elif args.plot == 'mi':
			if args.metric == 'adv':
				if alpha == None:
					plt.errorbar(EPSILONS, np.mean(adv_y_mi_1, axis=1), yerr=np.std(adv_y_mi_1, axis=1), color=str(color+0.4), fmt='-', capsize=2, label='Yeom MI 1')
					plt.errorbar(EPSILONS, np.mean(adv_y_mi_2, axis=1), yerr=np.std(adv_y_mi_2, axis=1), color=str(color+0.4), fmt='-', capsize=2, label='Yeom MI 2')
				plt.errorbar(EPSILONS, np.mean(adv_p_mi_1, axis=1), yerr=np.std(adv_p_mi_1, axis=1), color=str(color), fmt='-', capsize=2, label='MI 1')
				plt.errorbar(EPSILONS, np.mean(adv_p_mi_2, axis=1), yerr=np.std(adv_p_mi_2, axis=1), color=str(color), fmt='-', capsize=2, label='MI 2')
			elif args.metric == 'ppv':
				if alpha == None:
					plt.errorbar(EPSILONS, np.mean(ppv_y_mi_1, axis=1), yerr=np.std(ppv_y_mi_1, axis=1), color=str(color+0.4), fmt='-.', capsize=2, label='Yeom MI 1')
					plt.errorbar(EPSILONS, np.mean(ppv_y_mi_2, axis=1), yerr=np.std(ppv_y_mi_2, axis=1), color=str(color+0.4), fmt='-.', capsize=2, label='Yeom MI 2')
				plt.errorbar(EPSILONS, np.mean(ppv_p_mi_1, axis=1), yerr=np.std(ppv_p_mi_1, axis=1), color=str(color), fmt='-.', capsize=2, label='MI 1')
				plt.errorbar(EPSILONS, np.mean(ppv_p_mi_2, axis=1), yerr=np.std(ppv_p_mi_2, axis=1), color=str(color), fmt='-.', capsize=2, label='MI 2')
		elif args.plot == 'ai':
			if args.metric == 'adv':
				if alpha == None:
					plt.errorbar(EPSILONS, np.mean(adv_y_ai_1, axis=1), yerr=np.std(adv_y_ai_1, axis=1), color=str(color+0.4), fmt='-', capsize=2, label='Yeom AI 1')
					plt.errorbar(EPSILONS, np.mean(adv_y_ai_2, axis=1), yerr=np.std(adv_y_ai_2, axis=1), color=str(color+0.4), fmt='-', capsize=2, label='Yeom AI 2')
				plt.errorbar(EPSILONS, np.mean(adv_p_ai_1, axis=1), yerr=np.std(adv_p_ai_1, axis=1), color=str(color), fmt='-', capsize=2, label='MI 1')
				plt.errorbar(EPSILONS, np.mean(adv_p_ai_2, axis=1), yerr=np.std(adv_p_ai_2, axis=1), color=str(color), fmt='-', capsize=2, label='MI 2')
			elif args.metric == 'ppv':
				if alpha == None:
					plt.errorbar(EPSILONS, np.mean(ppv_y_ai_1, axis=1), yerr=np.std(ppv_y_ai_1, axis=1), color=str(color+0.4), fmt='-.', capsize=2, label='Yeom AI 1')
					plt.errorbar(EPSILONS, np.mean(ppv_y_ai_2, axis=1), yerr=np.std(ppv_y_ai_2, axis=1), color=str(color+0.4), fmt='-.', capsize=2, label='Yeom AI 2')
				plt.errorbar(EPSILONS, np.mean(ppv_p_ai_1, axis=1), yerr=np.std(ppv_p_ai_1, axis=1), color=str(color), fmt='-.', capsize=2, label='MI 1')
				plt.errorbar(EPSILONS, np.mean(ppv_p_ai_2, axis=1), yerr=np.std(ppv_p_ai_2, axis=1), color=str(color), fmt='-.', capsize=2, label='MI 2')
		color += 0.2

	if args.plot == 'mi':
		yeom_1_adv = adv_y_mi_1
		yeom_2_adv = adv_y_mi_2
		our_1_adv = adv_p_mi_1
		our_2_adv = adv_p_mi_2
		yeom_1_ppv = ppv_y_mi_1
		yeom_2_ppv = ppv_y_mi_2
		our_1_ppv = ppv_p_mi_1
		our_2_ppv = ppv_p_mi_2
		yeom_1_label = "Yeom MI 1"
		yeom_2_label = "Yeom MI 2"
		our_1_label = "MI 1"
		our_2_label = "MI 2"
	elif args.plot == 'ai':
		yeom_1_adv = adv_y_ai_1
		yeom_2_adv = adv_y_ai_2
		our_1_adv = adv_p_ai_1
		our_2_adv = adv_p_ai_2
		yeom_1_ppv = ppv_y_ai_1
		yeom_2_ppv = ppv_y_ai_2
		our_1_ppv = ppv_p_ai_1
		our_2_ppv = ppv_p_ai_2
		yeom_1_label = "Yeom AI 1"
		yeom_2_label = "Yeom AI 2"
		our_1_label = "AI 1"
		our_2_label = "AI 2"
	plt.xscale('log')
	plt.xlabel('Privacy Budget ($\epsilon$)')	
	if args.plot == 'acc':
		plt.ylabel('Accuracy Loss')
		plt.yticks(np.arange(0, 1.1, step=0.2))
		plt.annotate("RDP", pretty_position(EPSILONS, y["rdp_"], 2), textcoords="offset points", xytext=(20, 10), ha='right', color=str(0.3))
		plt.annotate("GDP", pretty_position(EPSILONS, y["gdp_"], 2), textcoords="offset points", xytext=(-20, -10), ha='right', color=str(0.1))
		plt.tight_layout()
	else:
		bottom, top = plt.ylim()
		if args.metric == 'adv':
			if alpha == None:
				plt.errorbar(EPS, improved_limit(EPS), color='black', fmt='--', capsize=2, label='Improved Limit')
				plt.annotate("$Adv_\mathcal{A}$ Bound", pretty_position(EPS, improved_limit(EPS), 50), textcoords="offset points", xytext=(0,-20), ha='left')
				plt.annotate(yeom_1_label, pretty_position(EPSILONS, np.mean(yeom_1_adv, axis=1), 0), textcoords="offset points", xytext=(-40,20), ha='left', color=str(0.5))
				plt.annotate(yeom_2_label, pretty_position(EPSILONS, np.mean(yeom_2_adv, axis=1), 4), textcoords="offset points", xytext=(-20,-30), ha='left', color=str(0.5))
			else:
				plt.errorbar(EPS, [adv_lim(eps, delta=delta, alpha=alpha) for eps in EPS], color='black', fmt='--', capsize=2, label='Improved Limit')
				plt.annotate("$Adv_\mathcal{A}$ Bound", pretty_position(EPS, [adv_lim(eps, delta=delta, alpha=alpha) for eps in EPS], 100), textcoords="offset points", xytext=(-5,0), ha='right')
			plt.ylim(0, 0.3)
			plt.yticks(np.arange(0, 0.31, step=0.05))
			plt.annotate(our_1_label, pretty_position(EPSILONS, np.mean(our_1_adv, axis=1), 3), textcoords="offset points", xytext=(-10,20), ha='left', color=str(0.1))
			plt.annotate(our_2_label, pretty_position(EPSILONS, np.mean(our_2_adv, axis=1), 4), textcoords="offset points", xytext=(0,20), ha='left', color=str(0.1))	
			plt.ylabel('$Adv_\mathcal{A}$')
		elif args.metric == 'ppv':
			if alpha == None:
				plt.annotate(yeom_1_label, pretty_position(EPSILONS, np.mean(yeom_1_ppv, axis=1), 0), textcoords="offset points", xytext=(-40,20), ha='left', color=str(0.5))
				plt.annotate(yeom_2_label, pretty_position(EPSILONS, np.mean(yeom_2_ppv, axis=1), 4), textcoords="offset points", xytext=(-20,-20), ha='left', color=str(0.5))
			else:
				plt.errorbar(EPS, [ppv_lim(eps, delta=delta, alpha=alpha) for eps in EPS], color='black', fmt='--', capsize=2, label='Improved Limit')
				plt.annotate("$PPV_\mathcal{A}$ Bound", pretty_position(EPS, [ppv_lim(eps, delta=delta, alpha=alpha) for eps in EPS], 30), textcoords="offset points", xytext=(5,0), ha='left')
			plt.ylim(0.5, 0.62)
			plt.yticks(np.arange(0.5, 0.63, step=0.02))
			plt.annotate(our_1_label, pretty_position(EPSILONS, np.mean(our_1_ppv, axis=1), 3), textcoords="offset points", xytext=(-10,20), ha='left', color=str(0.1))
			plt.annotate(our_2_label, pretty_position(EPSILONS, np.mean(our_2_ppv, axis=1), 4), textcoords="offset points", xytext=(0,-20), ha='left', color=str(0.1))	
			plt.ylabel('$PPV_\mathcal{A}$')
		plt.tight_layout()
	plt.show()
	


def members_revealed_fixed_fpr(result):
	thres = args.fpr_threshold# 0.01 == 1% FPR, 0.02 == 2% FPR, 0.05 == 5% FPR
	_, _, train_loss, membership, _, attack_pred, _, mem_pred, _, attr_mem, attr_pred, _ = pickle.load(open(DATA_PATH+MODEL+'no_privacy_'+str(args.l2_ratio)+'.p', 'rb'))
	pred = (max(mem_pred) - mem_pred) / (max(mem_pred) - min(mem_pred))
	#pred = attack_pred[:,1]
	print(len(_members_revealed(membership, pred, thres)))
	for dp in DP:
		for eps in EPSILONS:
			mems_revealed = []
			for run in RUNS:
				_, _, train_loss, membership, _, attack_pred, _, mem_pred, _, attr_mem, attr_pred, _ = result[dp][eps][run]
				pred = (max(mem_pred) - mem_pred) / (max(mem_pred) - min(mem_pred))
				#pred = attack_pred[:,1]
				mems_revealed.append(_members_revealed(membership, pred, thres))
			s = set.intersection(*mems_revealed)
			print(dp, eps, len(s))


def _members_revealed(membership, prediction, acceptable_fpr):
	fpr, tpr, thresholds = roc_curve(membership, prediction, pos_label=1)
	l = list(filter(lambda x: x < acceptable_fpr, fpr))
	if len(l) == 0:
		print("Error: low acceptable fpr")
		return None
	threshold = thresholds[len(l)-1]
	preds = list(map(lambda val: 1 if val >= threshold else 0, prediction))
	tp = [a*b for a,b in zip(preds,membership)]
	revealed = list(map(lambda i: i if tp[i] == 1 else None, range(len(tp))))
	return set(list(filter(lambda x: x != None, revealed)))


def ppv_across_runs(mem, pred):
	tn, fp, fn, tp = confusion_matrix(mem, np.where(pred >= 0, 1, 0)).ravel()
	print("0 or more")
	print(tp, fp, tp / (tp + fp))
	tn, fp, fn, tp = confusion_matrix(mem, np.where(pred >= 1, 1, 0)).ravel()
	print("1 or more")
	print(tp, fp, tp / (tp + fp))
	tn, fp, fn, tp = confusion_matrix(mem, np.where(pred >= 2, 1, 0)).ravel()
	print("2 or more")
	print(tp, fp, tp / (tp + fp))
	tn, fp, fn, tp = confusion_matrix(mem, np.where(pred >= 3, 1, 0)).ravel()
	print("3 or more")
	print(tp, fp, tp / (tp + fp))
	tn, fp, fn, tp = confusion_matrix(mem, np.where(pred >= 4, 1, 0)).ravel()
	print("4 or more")
	print(tp, fp, tp / (tp + fp))
	tn, fp, fn, tp = confusion_matrix(mem, np.where(pred == 5, 1, 0)).ravel()
	print("exactly 5")
	print(tp, fp, tp / (tp + fp))


def members_revealed_fixed_threshold(result):
	_, _, train_loss, membership, attack_adv, attack_pred, mem_adv, mem_pred, attr_adv, attr_mem, attr_pred, _ = pickle.load(open(DATA_PATH+MODEL+'no_privacy_'+str(args.l2_ratio)+'.p', 'rb'))
	print(attack_adv, mem_adv, np.mean(attr_adv))
	pred = np.where(mem_pred > train_loss, 0, 1)
	#pred = np.where(attack_pred[:,1] <= 0.5, 0, 1)
	#attr_pred = np.array(attr_pred)
	#membership = np.array(attr_mem).ravel()
	#pred = np.where(stats.norm(0, train_loss).pdf(attr_pred[:,0,:]) >= stats.norm(0, train_loss).pdf(attr_pred[:,1,:]), 0, 1).ravel()
	tn, fp, fn, tp = confusion_matrix(membership, pred).ravel()
	print(tp, tp / (tp + fp))
	fpr, tpr, thresholds = roc_curve(membership, pred, pos_label=1)
	print(fpr, tpr, np.max(tpr-fpr))
	
	for dp in DP:
		for eps in EPSILONS:
			ppv, preds = [], []
			for run in RUNS:
				_, _, train_loss, membership, _, attack_pred, _, mem_pred, _, attr_mem, attr_pred, _ = result[dp][eps][run]
				pred = np.where(mem_pred > train_loss, 0, 1)
				preds.append(pred)				
				#pred = np.where(attack_pred[:,1] <= 0.5, 0, 1)
				#attr_pred = np.array(attr_pred)
				#membership = np.array(attr_mem).ravel()
				#pred = np.where(stats.norm(0, train_loss).pdf(attr_pred[:,0,:]) >= stats.norm(0, train_loss).pdf(attr_pred[:,1,:]), 0, 1).ravel()
				ppv.append(get_ppv(membership, pred))
			print(dp, eps, np.mean(ppv))
			preds = np.sum(np.array(preds), axis=0)
			ppv_across_runs(membership, preds)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset', type=str)
	parser.add_argument('--model', type=str, default='nn')
	parser.add_argument('--l2_ratio', type=str, default='1e-08')
	parser.add_argument('--gamma', type=int, default=1)
	parser.add_argument('--alpha', type=float, default=None)
	parser.add_argument('--function', type=int, default=1)
	parser.add_argument('--plot', type=str, default='acc')
	parser.add_argument('--metric', type=str, default='adv')
	parser.add_argument('--fpr_threshold', type=float, default=0.01)
	parser.add_argument('--silent', type=int, default=1)
	args = parser.parse_args()
	print(vars(args))

	gamma = args.gamma
	alpha = args.alpha
	DATA_PATH = 'results/' + str(args.dataset) + '_improved_mi/'
	MODEL = str(gamma) + '_' + str(args.model) + '_'

	result = get_data()
	if args.function == 1:
		generate_plots(result) # plot the utility and privacy loss graphs
	elif args.function == 2:
		members_revealed_fixed_fpr(result) # return the number of members revealed for different FPR rates
	else:
		members_revealed_fixed_threshold(result)
