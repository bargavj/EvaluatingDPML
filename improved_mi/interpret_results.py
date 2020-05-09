from sklearn.metrics import roc_curve
from ..code.utilities import get_fp_adv_ppv, get_ppv, get_inference_threshold, plot_histogram, plot_sign_histogram
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse


EPS = list(np.arange(0.1, 100, 0.01))
EPS2 = list(np.arange(0.1, 100, 0.01))
EPSILONS = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
PERTURBATION = 'grad_pert_'
DP = ['gdp_', 'rdp_']
TYPE = ['o-', '.-']
DP_LABELS = ['GDP', 'RDP']
RUNS = range(5)
A, B = len(EPSILONS), len(RUNS)
ALPHAS = np.arange(0.01, 1, 0.01)
delta = 1e-5

new_rc_params = {
	'font.size': 18,
	'text.usetex': True,
	'font.family': 'Times New Roman',
	'mathtext.fontset': 'stix',
	'xtick.major.pad': '8'
}
plt.rcParams.update(new_rc_params)

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


def get_pred_mem_mi(per_instance_loss, proposed_mi_outputs, method=1, fpr_threshold=None, per_class_thresh=False, fixed_thresh=False):
	true_y, v_true_y, v_membership, v_per_instance_loss, v_counts, counts = proposed_mi_outputs
	if method == 1:
		if per_class_thresh:
			classes = np.unique(true_y)
			pred_membership = np.zeros(len(v_membership))
			threshs = []
			for c in classes:
				c_indices = np.arange(len(true_y))[true_y == c]
				v_c_indices = np.arange(len(v_true_y))[v_true_y == c]
				if fixed_thresh:
					thresh = np.mean(v_per_instance_loss[list(filter(lambda i: i < 10000, v_c_indices))]) 
				else:
					thresh = -get_inference_threshold(-v_per_instance_loss[v_c_indices], v_membership[v_c_indices], fpr_threshold)
				pred_membership[c_indices] = np.where(per_instance_loss[c_indices] <= thresh, 1, 0)
				threshs.append(thresh)
			print(max(0, min(threshs)), max(0, np.median(threshs)), max(0, max(threshs)))
			#plt.yscale('log')
			#plt.ylim(1e-6, 1e1)
			#plt.plot(list(range(1, 101)), list(map(lambda x: -x, threshs)))
			#plt.show()
			return max(0, threshs[0]), pred_membership
		else:
			thresh = -get_inference_threshold(-v_per_instance_loss, v_membership, fpr_threshold)
			return max(0, thresh), np.where(per_instance_loss <= thresh, 1, 0)
	else:
		if per_class_thresh:
			classes = np.unique(true_y)
			pred_membership = np.zeros(len(v_membership))
			threshs = []
			for c in classes:
				c_indices = np.arange(len(true_y))[true_y == c]
				v_c_indices = np.arange(len(v_true_y))[v_true_y == c]
				thresh = get_inference_threshold(v_counts[v_c_indices], v_membership[v_c_indices], fpr_threshold)
				pred_membership[c_indices] = np.where(counts[c_indices] >= thresh, 1, 0)
				threshs.append(thresh)
			#print(min(threshs), np.median(threshs), max(threshs))
			#plt.plot(list(range(1, 101)), threshs)
			#plt.ylim(0, 100)
			#plt.show()
			return threshs[0], pred_membership
		else:
			thresh = get_inference_threshold(v_counts, v_membership, fpr_threshold)
			return thresh, np.where(counts >= thresh, 1, 0)


def get_pred_mem_ai(per_instance_loss, proposed_mi_outputs, proposed_ai_outputs, i, method=1, fpr_threshold=None):
	true_y, v_true_y, v_membership, v_per_instance_loss, v_counts, counts = proposed_mi_outputs
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
	return thresh, mask & (pred_attribute_value ^ true_attribute_value_all[i] ^ [1]*len(pred_attribute_value))


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
		ax1.annotate('$Adv_\mathcal{A}$', pretty_position(phi, Adv_A, np.argmax(Adv_A)), textcoords="offset points", xytext=(-5,10), ha='right')
		ax1.annotate('$PPV_\mathcal{A}$', pretty_position(phi, PPV_A, -50), textcoords="offset points", xytext=(-20,20), ha='left')
		ax2.annotate('FPR ($\\alpha$)', pretty_position(phi, fpr, 0), textcoords="offset points", xytext=(-20,-10), ha='right')
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


def analyse_most_vulnerable(values, membership, top_k=1, reverse=False):
	vals = sorted(list(zip(values, membership, list(range(len(membership))))), key=(lambda x:x[0]), reverse=reverse)
	vul_dict = {}
	for val in vals:
		if len(vul_dict) > top_k:
			break
		if val[0] not in vul_dict:
			vul_dict[val[0]] = {0:[], 1:[]}
		vul_dict[val[0]][val[1]].append(val[2])
		dummy_key = val[0]
	del vul_dict[dummy_key]
	for key in vul_dict:
		print(key, len(vul_dict[key][1]), len(vul_dict[key][0]))
		print('')


def generate_plots(result):
	train_accs, baseline_acc = np.zeros(B), np.zeros(B)
	adv_y_mi_1, adv_y_mi_2, adv_p_mi_1, adv_p_mi_2 = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
	ppv_y_mi_1, ppv_y_mi_2, ppv_p_mi_1, ppv_p_mi_2 = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
	fpr_y_mi_1, fpr_y_mi_2, fpr_p_mi_1, fpr_p_mi_2 = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
	thresh_y_mi_1, thresh_y_mi_2, thresh_p_mi_1, thresh_p_mi_2 = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
	mi_1_zero_m, mi_1_zero_nm, mi_2_zero_m, mi_2_zero_nm = [], [], [], []
	for run in RUNS:
		aux, membership, per_instance_loss, yeom_mi_outputs_1, yeom_mi_outputs_2, proposed_mi_outputs = result['no_privacy'][run]
		train_loss, train_acc, test_loss, test_acc = aux
		true_y, v_true_y, v_membership, v_per_instance_loss, v_counts, counts = proposed_mi_outputs
		m, nm = 0, 0
		for i, val in enumerate(per_instance_loss):
			if val == 0:
				if membership[i] == 1:
					m += 1
				else:
					nm += 1
		mi_1_zero_m.append(m)
		mi_1_zero_nm.append(nm)
		m, nm = 0, 0
		for i, val in enumerate(counts):
			if val == 0:
				if membership[i] == 1:
					m += 1
				else:
					nm += 1
		mi_2_zero_m.append(m)
		mi_2_zero_nm.append(nm)
		#print(np.mean(counts[:10000]), np.std(counts[:10000]))
		#print(np.mean(counts[10000:]), np.std(counts[10000:]))
		#plot_histogram(per_instance_loss)
		#plot_distributions(per_instance_loss, membership)
		#plot_sign_histogram(membership, counts, 100)
		#plot_distributions(counts, membership, 2)
		#analyse_most_vulnerable(per_instance_loss, membership, top_k=5)
		#analyse_most_vulnerable(counts, membership, top_k=5, reverse=True)
		baseline_acc[run] = test_acc
		train_accs[run] = train_acc
		
		thresh, pred = get_pred_mem_mi(per_instance_loss, proposed_mi_outputs, method=1, fpr_threshold=alpha, per_class_thresh=args.per_class_thresh, fixed_thresh=args.fixed_thresh)
		fp, adv, ppv = get_fp_adv_ppv(membership, pred)
		thresh_p_mi_1[run], fpr_p_mi_1[run], adv_p_mi_1[run], ppv_p_mi_1[run] = thresh, fp / (gamma * 10000), adv, ppv
		thresh, pred = get_pred_mem_mi(per_instance_loss, proposed_mi_outputs, method=2, fpr_threshold=alpha, per_class_thresh=args.per_class_thresh, fixed_thresh=args.fixed_thresh)
		fp, adv, ppv = get_fp_adv_ppv(membership, pred)
		thresh_p_mi_2[run], fpr_p_mi_2[run], adv_p_mi_2[run], ppv_p_mi_2[run] = thresh, fp / (gamma * 10000), adv, ppv
		fp, adv, ppv = get_fp_adv_ppv(membership, yeom_mi_outputs_1)
		thresh_y_mi_1[run], fpr_y_mi_1[run], adv_y_mi_1[run], ppv_y_mi_1[run] = train_loss, fp / (gamma * 10000), adv, ppv
		#fp, adv, ppv = get_fp_adv_ppv(membership, yeom_mi_outputs_2)
		#fpr_y_mi_2[run], adv_y_mi_2[run], ppv_y_mi_2[run] = thresh, fp / (gamma * 10000), adv, ppv

	baseline_acc = np.mean(baseline_acc)
	print(np.mean(train_accs), baseline_acc)
	print('\nMI 1: \t %.2f +/- %.2f \t %.2f +/- %.2f' % (np.mean(mi_1_zero_m), np.std(mi_1_zero_m), np.mean(mi_1_zero_nm), np.std(mi_1_zero_nm)))
	print('\nMI 2: \t %.2f +/- %.2f \t %.2f +/- %.2f' % (np.mean(mi_2_zero_m), np.std(mi_2_zero_m), np.mean(mi_2_zero_nm), np.std(mi_2_zero_nm)))
	print('\nYeom MI 1:\nphi: %f +/- %f\nFPR: %.4f +/- %.4f\nTPR: %.4f +/- %.4f\nAdv: %.4f +/- %.4f\nPPV: %.4f +/- %.4f' % (np.mean(thresh_y_mi_1), np.std(thresh_y_mi_1), np.mean(fpr_y_mi_1), np.std(fpr_y_mi_1), np.mean(adv_y_mi_1+fpr_y_mi_1), np.std(adv_y_mi_1+fpr_y_mi_1), np.mean(adv_y_mi_1), np.std(adv_y_mi_1), np.mean(ppv_y_mi_1), np.std(ppv_y_mi_1)))
	#print('Yeom MI 2:\n TP: %d, Adv: %f, PPV: %f' % (np.mean(tp_y_mi_2), np.mean(adv_y_mi_2), np.mean(ppv_y_mi_2)))
	print('\nProposed MI 1:\nphi: %f +/- %f\nFPR: %.4f +/- %.4f\nTPR: %.4f +/- %.4f\nAdv: %.4f +/- %.4f\nPPV: %.4f +/- %.4f' % (np.mean(thresh_p_mi_1), np.std(thresh_p_mi_1), np.mean(fpr_p_mi_1), np.std(fpr_p_mi_1), np.mean(adv_p_mi_1+fpr_p_mi_1), np.std(adv_p_mi_1+fpr_p_mi_1), np.mean(adv_p_mi_1), np.std(adv_p_mi_1), np.mean(ppv_p_mi_1), np.std(ppv_p_mi_1)))
	print('\nProposed MI 2:\nphi: %f +/- %f\nFPR: %.4f +/- %.4f\nTPR: %.4f +/- %.4f\nAdv: %.4f +/- %.4f\nPPV: %.4f +/- %.4f' % (np.mean(thresh_p_mi_2), np.std(thresh_p_mi_2), np.mean(fpr_p_mi_2), np.std(fpr_p_mi_2), np.mean(adv_p_mi_2+fpr_p_mi_2), np.std(adv_p_mi_2+fpr_p_mi_2), np.mean(adv_p_mi_2), np.std(adv_p_mi_2), np.mean(ppv_p_mi_2), np.std(ppv_p_mi_2)))

    color = 0.1
	y = dict()
	for dp in DP:
		test_acc_vec = np.zeros((A, B))
		adv_y_mi_1, adv_y_mi_2, adv_p_mi_1, adv_p_mi_2 = np.zeros((A, B)), np.zeros((A, B)), np.zeros((A, B)), np.zeros((A, B))
		ppv_y_mi_1, ppv_y_mi_2, ppv_p_mi_1, ppv_p_mi_2 = np.zeros((A, B)), np.zeros((A, B)), np.zeros((A, B)), np.zeros((A, B))
		fpr_y_mi_1, fpr_y_mi_2, fpr_p_mi_1, fpr_p_mi_2 = np.zeros((A, B)), np.zeros((A, B)), np.zeros((A, B)), np.zeros((A, B))
		thresh_y_mi_1, thresh_y_mi_2, thresh_p_mi_1, thresh_p_mi_2 = np.zeros((A, B)), np.zeros((A, B)), np.zeros((A, B)), np.zeros((A, B))
		mi_1_zero_m, mi_1_zero_nm, mi_2_zero_m, mi_2_zero_nm = [], [], [], []
		for a, eps in enumerate(EPSILONS):
			for run in RUNS:
				aux, membership, per_instance_loss, yeom_mi_outputs_1, yeom_mi_outputs_2, proposed_mi_outputs = result[dp][eps][run]
				train_loss, train_acc, test_loss, test_acc = aux
				true_y, v_true_y, v_membership, v_per_instance_loss, v_counts, counts = proposed_mi_outputs
				test_acc_vec[a, run] = test_acc
				m, nm = 0, 0
				for i, val in enumerate(per_instance_loss):
					if val == 0:
						if membership[i] == 1:
							m += 1
						else:
							nm += 1
				mi_1_zero_m.append(m)
				mi_1_zero_nm.append(nm)
				m, nm = 0, 0
				for i, val in enumerate(counts):
					if val == 0:
						if membership[i] == 1:
							m += 1
						else:
							nm += 1
				mi_2_zero_m.append(m)
				mi_2_zero_nm.append(nm)
				#print(np.mean(counts[:10000]), np.std(counts[:10000]))
				#print(np.mean(counts[10000:]), np.std(counts[10000:]))
				#print(eps, run)
				#plot_histogram(per_instance_loss)
				#plot_distributions(per_instance_loss, membership)
				#plot_sign_histogram(membership, counts, 100)
				#plot_distributions(counts, membership, 2)
				#analyse_most_vulnerable(per_instance_loss, membership, top_k=5)
				#analyse_most_vulnerable(counts, membership, top_k=5, reverse=True)
				thresh, pred = get_pred_mem_mi(per_instance_loss, proposed_mi_outputs, method=1, fpr_threshold=alpha, per_class_thresh=args.per_class_thresh, fixed_thresh=args.fixed_thresh)
				fp, adv, ppv = get_fp_adv_ppv(membership, pred)
				thresh_p_mi_1[a, run], fpr_p_mi_1[a, run], adv_p_mi_1[a, run], ppv_p_mi_1[a, run] = thresh, fp / (gamma * 10000), adv, ppv
				thresh, pred = get_pred_mem_mi(per_instance_loss, proposed_mi_outputs, method=2, fpr_threshold=alpha, per_class_thresh=args.per_class_thresh, fixed_thresh=args.fixed_thresh)
				fp, adv, ppv = get_fp_adv_ppv(membership, pred)
				thresh_p_mi_2[a, run], fpr_p_mi_2[a, run], adv_p_mi_2[a, run], ppv_p_mi_2[a, run] = thresh, fp / (gamma * 10000), adv, ppv
				fp, adv, ppv = get_fp_adv_ppv(membership, yeom_mi_outputs_1)
				thresh_y_mi_1[a, run], fpr_y_mi_1[a, run], adv_y_mi_1[a, run], ppv_y_mi_1[a, run] = train_loss, fp / (gamma * 10000), adv, ppv
				#fp, adv, ppv = get_fp_adv_ppv(membership, yeom_mi_outputs_2)
				#thresh_y_mi_2[a, run], fpr_y_mi_2[a, run], adv_y_mi_2[a, run], ppv_y_mi_2[a, run] = thresh, fp / (gamma * 10000), adv, ppv
				
			print('\n'+str(eps)+'\n')
			print('\nMI 1: \t %.2f +/- %.2f \t %.2f +/- %.2f' % (np.mean(mi_1_zero_m), np.std(mi_1_zero_m), np.mean(mi_1_zero_nm), np.std(mi_1_zero_nm)))
			print('\nMI 2: \t %.2f +/- %.2f \t %.2f +/- %.2f' % (np.mean(mi_2_zero_m), np.std(mi_2_zero_m), np.mean(mi_2_zero_nm), np.std(mi_2_zero_nm)))
			print('\nYeom MI 1:\nphi: %f +/- %f\nFPR: %.4f +/- %.4f\nTPR: %.4f +/- %.4f\nAdv: %.4f +/- %.4f\nPPV: %.4f +/- %.4f' % (np.mean(thresh_y_mi_1[a]), np.std(thresh_y_mi_1[a]), np.mean(fpr_y_mi_1[a]), np.std(fpr_y_mi_1[a]), np.mean(adv_y_mi_1[a]+fpr_y_mi_1[a]), np.std(adv_y_mi_1[a]+fpr_y_mi_1[a]), np.mean(adv_y_mi_1[a]), np.std(adv_y_mi_1[a]), np.mean(ppv_y_mi_1[a]), np.std(ppv_y_mi_1[a])))
			print('\nProposed MI 1:\nphi: %f +/- %f\nFPR: %.4f +/- %.4f\nTPR: %.4f +/- %.4f\nAdv: %.4f +/- %.4f\nPPV: %.4f +/- %.4f' % (np.mean(thresh_p_mi_1[a]), np.std(thresh_p_mi_1[a]), np.mean(fpr_p_mi_1[a]), np.std(fpr_p_mi_1[a]), np.mean(adv_p_mi_1[a]+fpr_p_mi_1[a]), np.std(adv_p_mi_1[a]+fpr_p_mi_1[a]), np.mean(adv_p_mi_1[a]), np.std(adv_p_mi_1[a]), np.mean(ppv_p_mi_1[a]), np.std(ppv_p_mi_1[a])))
			print('\nProposed MI 2:\nphi: %f +/- %f\nFPR: %.4f +/- %.4f\nTPR: %.4f +/- %.4f\nAdv: %.4f +/- %.4f\nPPV: %.4f +/- %.4f' % (np.mean(thresh_p_mi_2[a]), np.std(thresh_p_mi_2[a]), np.mean(fpr_p_mi_2[a]), np.std(fpr_p_mi_2[a]), np.mean(adv_p_mi_2[a]+fpr_p_mi_2[a]), np.std(adv_p_mi_2[a]+fpr_p_mi_2[a]), np.mean(adv_p_mi_2[a]), np.std(adv_p_mi_2[a]), np.mean(ppv_p_mi_2[a]), np.std(ppv_p_mi_2[a])))
			
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


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset', type=str)
	parser.add_argument('--model', type=str, default='nn')
	parser.add_argument('--l2_ratio', type=str, default='1e-08')
	parser.add_argument('--gamma', type=int, default=1)
	parser.add_argument('--alpha', type=float, default=None)
	parser.add_argument('--per_class_thresh', type=int, default=0)
	parser.add_argument('--fixed_thresh', type=int, default=0)
	parser.add_argument('--plot', type=str, default='acc')
	parser.add_argument('--metric', type=str, default='adv')
	args = parser.parse_args()
	print(vars(args))

	gamma = args.gamma
	alpha = args.alpha
	DATA_PATH = '../results/' + str(args.dataset)
	MODEL = str(gamma) + '_' + str(args.model) + '_'

	result = get_data()
	generate_plots(result)