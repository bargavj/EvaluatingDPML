import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import roc_curve
from sklearn.metrics import confusion_matrix
from matplotlib_venn import venn3

EPS = list(np.arange(0.01, 0.1, 0.01)) + list(np.arange(0.1, 1, 0.1))
EPSILONS = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
PERTURBATION = 'grad_pert_'
DP = ['dp_', 'adv_cmp_', 'zcdp_', 'rdp_']
TYPE = ['o-', '.-', '^-', '--']
DP_LABELS = ['NC', 'AC', 'zCDP', 'RDP']
RUNS = range(5)

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams.update({'font.size': 15})


def theoretical_limit(epsilons):
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
	return result


def pretty_position(X, Y, pos):
	return ((X[pos] + X[pos+1]) / 2, (Y[pos] + Y[pos+1]) / 2)


def plot_advantage(result):
	train_acc, baseline_acc, train_loss, membership, _, shokri_mem_confidence, _, per_instance_loss, _, per_instance_loss_all, _ = pickle.load(open(DATA_PATH+MODEL+'no_privacy_'+str(args.l2_ratio)+'.p', 'rb'))
	print(train_acc, baseline_acc)
	color = 0.1
	y = dict()
	for dp in DP:
		test_acc_mean, yeom_mem_adv_mean, yeom_attr_adv_mean, shokri_mem_adv_mean = [], [], [], []
		test_acc_std, yeom_mem_adv_std, yeom_attr_adv_std, shokri_mem_adv_std = [], [], [], []
		for eps in EPSILONS:
			test_acc_d, yeom_mem_adv_d, yeom_attr_adv_d, shokri_mem_adv_d = [], [], [], []
			for run in RUNS:
				train_acc, test_acc, train_loss, membership, shokri_mem_adv, shokri_mem_confidence, yeom_mem_adv, per_instance_loss, yeom_attr_adv, per_instance_loss, features = result[dp][eps][run]
				test_acc_d.append(test_acc)
				yeom_mem_adv_d.append(yeom_mem_adv) # adversary's advantage using membership inference attack of Yeom et al.
				shokri_mem_adv_d.append(shokri_mem_adv) # adversary's advantage using membership inference attack of Shokri et al.
				yeom_attr_adv_d.append(np.mean(yeom_attr_adv)) # adversary's advantage using attribute inference attack of Yeom et al.
			test_acc_mean.append(np.mean(test_acc_d))
			test_acc_std.append(np.std(test_acc_d))
			yeom_mem_adv_mean.append(np.mean(yeom_mem_adv_d))
			yeom_mem_adv_std.append(np.std(yeom_mem_adv_d))
			shokri_mem_adv_mean.append(np.mean(shokri_mem_adv_d))
			shokri_mem_adv_std.append(np.std(shokri_mem_adv_d))
			yeom_attr_adv_mean.append(np.mean(yeom_attr_adv_d))
			yeom_attr_adv_std.append(np.std(yeom_attr_adv_d))

			if args.silent == 0:
				if args.plot == 'acc':
					print(dp, eps, (baseline_acc - np.mean(test_acc_d)) / baseline_acc, np.std(test_acc_d))
				elif args.plot == 'shokri_mi':
					print(dp, eps, np.mean(shokri_mem_adv_d), np.std(shokri_mem_adv_d))
				elif args.plot == 'yeom_ai':
					print(dp, eps, np.mean(yeom_attr_adv_d), np.std(yeom_attr_adv_d))
				elif args.plot == 'yeom_mi':
					print(dp, eps, np.mean(yeom_mem_adv_d), np.std(yeom_mem_adv_d))
		if args.plot == 'acc':
			y[dp] = (baseline_acc - test_acc_mean) / baseline_acc
			plt.errorbar(EPSILONS, (baseline_acc - test_acc_mean) / baseline_acc, yerr=test_acc_std, color=str(color), fmt='.-', capsize=2, label=DP_LABELS[DP.index(dp)])
		elif args.plot == 'shokri_mi':
			y[dp] = shokri_mem_adv_mean
			plt.errorbar(EPSILONS, shokri_mem_adv_mean, yerr=shokri_mem_adv_std, color=str(color), fmt='.-', capsize=2, label=DP_LABELS[DP.index(dp)])
		elif args.plot == 'yeom_ai':
			y[dp] = yeom_attr_adv_mean
			plt.errorbar(EPSILONS, yeom_attr_adv_mean, yerr=yeom_attr_adv_std, color=str(color), fmt='.-', capsize=2, label=DP_LABELS[DP.index(dp)])
		elif args.plot == 'yeom_mi':
			y[dp] = yeom_mem_adv_mean
			plt.errorbar(EPSILONS, yeom_mem_adv_mean, yerr=yeom_mem_adv_std, color=str(color), fmt='.-', capsize=2, label=DP_LABELS[DP.index(dp)])
		color += 0.2

	plt.xscale('log')
	plt.xlabel('Privacy Budget ($\epsilon$)')

	if args.plot == 'acc':
		plt.ylabel('Accuracy Loss')
		plt.yticks(np.arange(0, 1.1, step=0.2))
	else:
		bottom, top = plt.ylim()
		plt.errorbar(EPS, theoretical_limit(EPS), color='black', fmt='--', capsize=2, label='Theoretical Limit')
		plt.ylim(bottom, 0.25)
		plt.annotate("$\epsilon$-DP Bound", pretty_position(EPS, theoretical_limit(EPS), 9), textcoords="offset points", xytext=(5,0), ha='left')
		plt.yticks(np.arange(0, 0.26, step=0.05))
		plt.ylabel('Privacy Leakage')

	plt.annotate("RDP", pretty_position(EPSILONS, y["rdp_"], 8), textcoords="offset points", xytext=(-10, 0), ha='right')
	plt.annotate("zCDP", pretty_position(EPSILONS, y["zcdp_"], 7), textcoords="offset points", xytext=(8, 12), ha='right')
	plt.annotate("AC", pretty_position(EPSILONS, y["adv_cmp_"], -4), textcoords="offset points", xytext=(0, -10), ha='left')
	plt.annotate("NC", pretty_position(EPSILONS, y["dp_"], -4), textcoords="offset points", xytext=(-10, 0), ha='right')

	plt.show()


def members_revealed_fixed_fpr(result):
	thres = args.fpr_threshold# 0.01 == 1% FPR, 0.02 == 2% FPR, 0.05 == 5% FPR
	_, _, train_loss, membership, _, shokri_mem_confidence, _, per_instance_loss, _, per_instance_loss_all, _ = pickle.load(open(DATA_PATH+MODEL+'no_privacy_'+str(args.l2_ratio)+'.p', 'rb'))
	pred = (max(per_instance_loss) - per_instance_loss) / (max(per_instance_loss) - min(per_instance_loss))
	#pred = shokri_mem_confidence[:,1]
	print(len(_members_revealed(membership, pred, thres)))
	for dp in DP:
		for eps in EPSILONS:
			mems_revealed = []
			for run in RUNS:
				_, _, train_loss, membership, _, shokri_mem_confidence, _, per_instance_loss, _, per_instance_loss_all, _ = result[dp][eps][run]
				pred = (max(per_instance_loss) - per_instance_loss) / (max(per_instance_loss) - min(per_instance_loss))
				#pred = shokri_mem_confidence[:,1]
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

def generate_venn(mem, preds):
	run1 = preds[0]
	run2 = preds[1]

	run1_tp = []
	run1_fp = []
	run2_tp = []
	run2_fp = []
	bothpos = []
	bothtruepos = []

	for index in range(len(mem)):
		if mem[index] == 0:
			if run1[index] == 1:
				run1_fp += [index]
			if run2[index] == 1:
				run2_fp += [index]
		else:  # mem(index) == 0
			if run1[index] == 1:
				run1_tp += [index]
			if run2[index] == 1:
				run2_tp += [index]

	run1pos = run1_fp + run1_tp
	run2pos = run2_fp + run2_tp

	for mem in run1pos:
		if mem in run2pos:
			bothpos += [mem]
	for mem in run1_tp:
		if mem in run2_tp:
			bothtruepos += [mem]

	s1 = len(run1_fp)
	s2 = len(run2_fp)
	s3 = len(bothpos) - len(bothtruepos)
	s4 = 0
	s5 = len(run1_tp)
	s6 = len(run2_tp)
	s7 = len(bothtruepos)

	venn3(subsets=(s1,s2,s3,s4,s5,s6,s7), set_labels=("Run 1", "Run 2", "TP"))
	plt.text(-0.70, 0.30, "FP")
	plt.text(0.61, 0.30, "FP")
	plt.show()


def members_revealed_fixed_threshold(result):
	_, _, train_loss, membership, shokri_mem_adv, shokri_mem_confidence, yeom_mem_adv, per_instance_loss, yeom_attr_adv, per_instance_loss_all, _ = pickle.load(open(DATA_PATH+MODEL+'no_privacy_'+str(args.l2_ratio)+'.p', 'rb'))
	print(shokri_mem_adv, yeom_mem_adv, np.mean(yeom_attr_adv))
	pred = np.where(per_instance_loss > train_loss, 0, 1)
	#pred = np.where(shokri_mem_confidence[:,1] <= 0.5, 0, 1)
	#attr_pred = np.array(per_instance_loss_all)
	#pred = np.where(stats.norm(0, train_loss).pdf(attr_pred[:,0,:]) >= stats.norm(0, train_loss).pdf(attr_pred[:,1,:]), 0, 1).ravel()
	tn, fp, fn, tp = confusion_matrix(membership, pred).ravel()
	print(tp, tp / (tp + fp))
	fpr, tpr, thresholds = roc_curve(membership, pred, pos_label=1)
	print(fpr, tpr, np.max(tpr-fpr))
	
	for dp in DP:
		for eps in EPSILONS:
			ppv, preds = [], []
			for run in RUNS:
				_, _, train_loss, membership, _, shokri_mem_confidence, _, per_instance_loss, _, per_instance_loss_all, _ = result[dp][eps][run]
				pred = np.where(per_instance_loss > train_loss, 0, 1)
				preds.append(pred)				
				#pred = np.where(shokri_mem_confidence[:,1] <= 0.5, 0, 1)
				#attr_pred = np.array(per_instance_loss_all)
				#pred = np.where(stats.norm(0, train_loss).pdf(attr_pred[:,0,:]) >= stats.norm(0, train_loss).pdf(attr_pred[:,1,:]), 0, 1).ravel()
				ppv.append(get_ppv(membership, pred))
			print(dp, eps, np.mean(ppv))
			sumpreds = np.sum(np.array(preds), axis=0)
			ppv_across_runs(membership, sumpreds)

	if args.venn == 1:
		generate_venn(membership, preds)


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset', type=str)
	parser.add_argument('--model', type=str, default='nn')
	parser.add_argument('--l2_ratio', type=float, default=1e-5)
	parser.add_argument('--function', type=int, default=1)
	parser.add_argument('--plot', type=str, default='acc')
	parser.add_argument('--fpr_threshold', type=float, default=0.01)
	parser.add_argument('--silent', type=int, default=1)
	parser.add_argument('--venn', type=int, default=0)
	args = parser.parse_args()
	print(vars(args))

	DATA_PATH = '../results/' + str(args.dataset) + '/'
	MODEL = str(args.model) + '_'

	result = get_data()
	if args.function == 1:
		plot_advantage(result) # plot the utility and privacy loss graphs
	elif args.function == 2:
		members_revealed_fixed_fpr(result) # return the number of members revealed for different FPR rates
	else:
		members_revealed_fixed_threshold(result)