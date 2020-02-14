from sklearn.metrics import roc_curve, confusion_matrix
from scipy import stats
from utilities import get_adv, get_ppv, get_inference_threshold
from attack import evaluate_proposed_membership_inference
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse


EPS = list(np.arange(0.1, 100, 0.01))
EPS2 = list(np.arange(0.1, 100, 0.01))
EPSILONS = [0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
PERTURBATION = 'grad_pert_'
DP = ['gdp_']
TYPE = ['o-', '.-']
DP_LABELS = ['GDP', 'RDP']
RUNS = range(5)
A, B = len(EPSILONS), len(RUNS)
ALPHAS = np.arange(0.01, 1, 0.01)
delta = 1e-5
ALPHA = 0.1
gamma = 1

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
			return np.where(per_instance_loss <= -thresh, 1, 0)
		else:
			thresh = get_inference_threshold(v_counts, v_membership, fpr_threshold)
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

def plot_distributions(pred_vector, true_vector):
	fpr, tpr, thresholds = roc_curve(true_vector, pred_vector, pos_label=1)
	fpr, tpr, thresholds = np.array(fpr), np.array(tpr), np.array(thresholds)
	PPV_A = tpr / (tpr + gamma * fpr)
	Adv_A = tpr - fpr
	phi = (thresholds - np.min(thresholds))/ (np.max(thresholds) - np.min(thresholds))
	fig, ax1 = plt.subplots()
	ax1.plot(phi, Adv_A, label="Adv", color='black')
	ax1.plot(phi, PPV_A, label="PPV", color='black')
	ax2 = ax1.twinx()
	ax2.plot(phi, fpr, label="FPR", color='black', linestyle='dashed')
	ax1.annotate('$Adv_\mathcal{A}$', pretty_position(phi, Adv_A, -2), textcoords="offset points", xytext=(0,10), ha='left')
	ax1.annotate('$PPV_\mathcal{A}$', pretty_position(phi, PPV_A, -2), textcoords="offset points", xytext=(0,10), ha='left')
	ax2.annotate('FPR ($\\alpha$)', pretty_position(phi, fpr, -2), textcoords="offset points", xytext=(0,-30), ha='left')
	ax1.set_xlabel('Decision Function $\Phi$')
	ax1.set_ylabel('Privacy Leakage Metrics')
	ax2.set_ylabel('False Positive Rate')
	ax1.set_yticks(np.arange(0, 1.1, step=0.2))
	ax2.set_yticks(np.arange(0, 1.1, step=0.2))
	fig.tight_layout()
	plt.show()


def plot_advantage(result):
	train_accs, baseline_acc = np.zeros(B), np.zeros(B)
	adv_y_mi_1, adv_y_mi_2, adv_y_ai_1, adv_y_ai_2, adv_p_mi_1, adv_p_mi_2, adv_p_ai_1, adv_p_ai_2 = np.zeros(B), np.zeros(B), np.zeros(5*B), np.zeros(5*B), np.zeros(B), np.zeros(B), np.zeros(5*B), np.zeros(5*B)
	ppv_y_mi_1, ppv_y_mi_2, ppv_y_ai_1, ppv_y_ai_2, ppv_p_mi_1, ppv_p_mi_2, ppv_p_ai_1, ppv_p_ai_2 = np.zeros(B), np.zeros(B), np.zeros(5*B), np.zeros(5*B), np.zeros(B), np.zeros(B), np.zeros(5*B), np.zeros(5*B)
	pred1, pred2, pred3, pred4 = [], [], [], []
	for run in RUNS:
		aux, membership, per_instance_loss, features, yeom_mi_outputs_1, yeom_mi_outputs_2, yeom_ai_outputs_1, yeom_ai_outputs_2, proposed_mi_outputs, proposed_ai_outputs = result['no_privacy'][run]
		train_loss, train_acc, test_loss, test_acc = aux
		v_membership, v_per_instance_loss, v_counts, counts = proposed_mi_outputs
		plot_distributions(-per_instance_loss, membership)
		plot_distributions(counts, membership)
		baseline_acc[run] = test_acc
		train_accs[run] = train_acc
		adv_p_mi_1[run] = get_adv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, method=1, fpr_threshold=ALPHA))
		adv_p_mi_2[run] = get_adv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, method=2, fpr_threshold=ALPHA))
		ppv_p_mi_1[run] = get_ppv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, method=1, fpr_threshold=ALPHA))
		ppv_p_mi_2[run] = get_ppv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, method=2, fpr_threshold=ALPHA))
		adv_y_mi_1[run] = get_adv(membership, yeom_mi_outputs_1)
		adv_y_mi_2[run] = get_adv(membership, yeom_mi_outputs_2)
		ppv_y_mi_1[run] = get_ppv(membership, yeom_mi_outputs_1)
		ppv_y_mi_2[run] = get_ppv(membership, yeom_mi_outputs_2)
		pred1.append(yeom_mi_outputs_1)
		pred2.append(yeom_mi_outputs_2)
		pred3.append(get_pred_mem(per_instance_loss, proposed_mi_outputs, method=1, fpr_threshold=ALPHA))
		pred4.append(get_pred_mem(per_instance_loss, proposed_mi_outputs, method=2, fpr_threshold=ALPHA))
	baseline_acc = np.mean(baseline_acc)
	print(np.mean(train_accs), baseline_acc)
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
		for a, eps in enumerate(EPSILONS):
			if eps > 100 and dp == 'gdp_':
				test_acc_vec = test_acc_vec[:a]
				continue
			pred1, pred2, pred3, pred4 = [], [], [], []
			for run in RUNS:
				aux, membership, per_instance_loss, features, yeom_mi_outputs_1, yeom_mi_outputs_2, yeom_ai_outputs_1, yeom_ai_outputs_2, proposed_mi_outputs, proposed_ai_outputs = result[dp][eps][run]
				train_loss, train_acc, test_loss, test_acc = aux
				v_membership, v_per_instance_loss, v_counts, counts = proposed_mi_outputs
				print('Epsilon: %.1f, run: %d, Method 1' % (eps, run+1))
				plot_distributions(-per_instance_loss, membership)
				print('Epsilon: %.1f, run: %d, Method 2' % (eps, run+1))
				plot_distributions(counts, membership)
				test_acc_vec[a, run] = test_acc
				adv_p_mi_1[a, run] = get_adv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, method=1, fpr_threshold=ALPHA))
				adv_p_mi_2[a, run] = get_adv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, method=2, fpr_threshold=ALPHA))
				ppv_p_mi_1[a, run] = get_ppv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, method=1, fpr_threshold=ALPHA))
				ppv_p_mi_2[a, run] = get_ppv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, method=2, fpr_threshold=ALPHA))
				'''
				for i in range(5):
					adv_y_ai_1[a, run*5 + i] = get_adv(membership, yeom_ai_outputs_1[i])
					adv_y_ai_2[a, run*5 + i] = get_adv(membership, yeom_ai_outputs_2[i])
					ppv_y_ai_1[a, run*5 + i] = get_ppv(membership, yeom_ai_outputs_1[i])
					ppv_y_ai_2[a, run*5 + i] = get_ppv(membership, yeom_ai_outputs_2[i])
					adv_p_ai_1[a, run*5 + i] = get_adv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, proposed_ai_outputs, i, method=1, fpr_threshold=ALPHA))
					adv_p_ai_2[a, run*5 + i] = get_adv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, proposed_ai_outputs, i, method=2, fpr_threshold=ALPHA))
					ppv_p_ai_1[a, run*5 + i] = get_ppv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, proposed_ai_outputs, i, method=1, fpr_threshold=ALPHA))
					ppv_p_ai_2[a, run*5 + i] = get_ppv(membership, get_pred_mem(per_instance_loss, proposed_mi_outputs, proposed_ai_outputs, i, method=2, fpr_threshold=ALPHA))
				'''
				adv_y_mi_1[a, run] = get_adv(membership, yeom_mi_outputs_1)
				adv_y_mi_2[a, run] = get_adv(membership, yeom_mi_outputs_2)
				ppv_y_mi_1[a, run] = get_ppv(membership, yeom_mi_outputs_1)
				ppv_y_mi_2[a, run] = get_ppv(membership, yeom_mi_outputs_2)
				pred1.append(yeom_mi_outputs_1)
				pred2.append(yeom_mi_outputs_2)
				pred3.append(get_pred_mem(per_instance_loss, proposed_mi_outputs, method=1, fpr_threshold=ALPHA))
				pred4.append(get_pred_mem(per_instance_loss, proposed_mi_outputs, method=2, fpr_threshold=ALPHA))
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
		if args.plot == 'acc':
			y[dp] = 1 - np.mean(test_acc_vec, axis=1) / baseline_acc
			plt.errorbar(EPSILONS, 1 - np.mean(test_acc_vec, axis=1) / baseline_acc, yerr=np.std(test_acc_vec, axis=1), color=str(color), fmt='.-', capsize=2, label=DP_LABELS[DP.index(dp)])
		elif args.plot == 'adv':
			y[dp] = adv_y_mi_1
			plt.errorbar(EPSILONS, np.mean(adv_y_mi_1, axis=1), yerr=np.std(adv_y_mi_1, axis=1), color=str(color+0.5), fmt='-.', capsize=2, label='Yeom\'s MI Attack 1')
			plt.errorbar(EPSILONS, np.mean(adv_y_mi_2, axis=1), yerr=np.std(adv_y_mi_2, axis=1), color=str(color), fmt='-.', capsize=2, label='Yeom\'s MI Attack 2')
			plt.errorbar(EPSILONS, np.mean(adv_p_mi_1, axis=1), yerr=np.std(adv_p_mi_1, axis=1), color=str(color+0.5), fmt='-', capsize=2, label='Proposed MI Attack 1')
			plt.errorbar(EPSILONS, np.mean(adv_p_mi_2, axis=1), yerr=np.std(adv_p_mi_2, axis=1), color=str(color), fmt='-', capsize=2, label='Proposed MI Attack 2')
		elif args.plot == 'ppv':
			y[dp] = ppv_y_mi_1
			plt.errorbar(EPSILONS, np.mean(ppv_y_mi_1, axis=1), yerr=np.std(ppv_y_mi_1, axis=1), color=str(color+0.5), fmt='-.', capsize=2, label='Yeom\'s MI Attack 1')
			plt.errorbar(EPSILONS, np.mean(ppv_y_mi_2, axis=1), yerr=np.std(ppv_y_mi_2, axis=1), color=str(color), fmt='-.', capsize=2, label='Yeom\'s MI Attack 2')
			plt.errorbar(EPSILONS, np.mean(ppv_p_mi_1, axis=1), yerr=np.std(ppv_p_mi_1, axis=1), color=str(color+0.5), fmt='-', capsize=2, label='Proposed MI Attack 1')
			plt.errorbar(EPSILONS, np.mean(ppv_p_mi_2, axis=1), yerr=np.std(ppv_p_mi_2, axis=1), color=str(color), fmt='-', capsize=2, label='Proposed MI Attack 2')
		color += 0.2

	plt.xscale('log')
	plt.xlabel('Privacy Budget ($\epsilon$)')

	if args.plot == 'acc':
		plt.ylabel('Accuracy Loss')
		plt.yticks(np.arange(0, 1.1, step=0.2))
		plt.annotate("RDP", pretty_position(EPSILONS, y["rdp_"], 3), textcoords="offset points", xytext=(30, 10), ha='right', color=str(0.3))
		plt.annotate("GDP", pretty_position(EPSILONS, y["gdp_"], 3), textcoords="offset points", xytext=(-10, -10), ha='right', color=str(0.1))
	elif args.plot == 'adv':
		bottom, top = plt.ylim()
		plt.errorbar(EPS, yeoms_limit(EPS), color='black', fmt='--', capsize=2, label='Old Theoretical Limit')
		plt.errorbar(EPS, improved_limit(EPS), color='black', fmt='--', capsize=2, label='Improved Limit')
		plt.ylim(bottom, 1)
		plt.annotate("Yeom's", pretty_position(EPS, yeoms_limit(EPS), 55), textcoords="offset points", xytext=(-70,0), ha='left')
		plt.annotate("Bound", pretty_position(EPS, yeoms_limit(EPS), 55), textcoords="offset points", xytext=(-70,-20), ha='left')
		plt.annotate("Improved Bound", pretty_position(EPS, improved_limit(EPS), 500), textcoords="offset points", xytext=(0,-20), ha='left')
		plt.annotate("MI-1", pretty_position(EPSILONS, np.mean(adv_p_mi_1, axis=1), 3), textcoords="offset points", xytext=(0,10), ha='left', color=str(0.6))
		plt.annotate("MI-2", pretty_position(EPSILONS, np.mean(adv_p_mi_2, axis=1), 4), textcoords="offset points", xytext=(0,-10), ha='left', color=str(0.1))
		plt.annotate("Baseline-1", pretty_position(EPSILONS, np.mean(adv_y_mi_1, axis=1), 0), textcoords="offset points", xytext=(0,10), ha='left', color=str(0.6))
		plt.annotate("Baseline-2", pretty_position(EPSILONS, np.mean(adv_y_mi_2, axis=1), 4), textcoords="offset points", xytext=(0,-10), ha='left', color=str(0.1))
		plt.yticks(np.arange(0, 1.1, step=0.2))
		plt.ylabel('$Adv_\mathcal{A}$')
	elif args.plot == 'ppv':
		bottom, top = plt.ylim()
		plt.errorbar(EPS, [ppv_lim(eps, delta=delta, alpha=ALPHA) for eps in EPS], color='black', fmt='--', capsize=2, label='Improved Limit')
		plt.ylim(bottom, 1)
		plt.annotate("$\epsilon$-DP Bound", pretty_position(EPS, [ppv_lim(eps, delta=delta, alpha=ALPHA) for eps in EPS], 1000), textcoords="offset points", xytext=(-10,-15), ha='left')
		plt.yticks(np.arange(0, 1.1, step=0.2))
		plt.ylabel('$PPV_\mathcal{A}$')
	
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


def get_ppv(mem, pred):
	tn, fp, fn, tp = confusion_matrix(mem, pred).ravel()
	return tp / (tp + fp)


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
	parser.add_argument('--l2_ratio', type=str, default='1e-5')
	parser.add_argument('--function', type=int, default=1)
	parser.add_argument('--plot', type=str, default='acc')
	parser.add_argument('--fpr_threshold', type=float, default=0.01)
	parser.add_argument('--silent', type=int, default=1)
	args = parser.parse_args()
	print(vars(args))

	DATA_PATH = 'results/' + str(args.dataset) + '_improved_mi/'
	MODEL = str(gamma) + '_' + str(args.model) + '_'

	result = get_data()
	if args.function == 1:
		plot_advantage(result) # plot the utility and privacy loss graphs
	elif args.function == 2:
		members_revealed_fixed_fpr(result) # return the number of members revealed for different FPR rates
	else:
		members_revealed_fixed_threshold(result)
