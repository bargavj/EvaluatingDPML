from sklearn.metrics import classification_report, accuracy_score, roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pickle

DATA_PATH = 'results/purchase_100/'
FILE_SUFFIX = '_nn_grad_pert_dp_1000.0.p'

EPS = list(np.arange(0.01, 0.1, 0.01)) + list(np.arange(0.1, 1, 0.1))
EPSILONS = [0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0, 100.0, 500.0, 1000.0]
MODEL = 'nn_'
PERTURBATION = 'grad_pert_'
DP = ['dp_', 'adv_cmp_', 'zcdp_', 'rdp_']
TYPE = ['o-', '.-', '^-', '--']
DP_LABELS = ['Naive Composition', 'Advanced Composition', 'zCDP', 'RDP']
RUNS = range(5)


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


def plot_advantage(result):
	train_acc, baseline_acc, train_loss, membership, _, attack_pred, _, mem_pred, _, attr_mem, attr_pred, _ = pickle.load(open(DATA_PATH+MODEL+'no_privacy_1e-8'+'.p', 'rb'))
	print(train_acc, baseline_acc)
	color = 0.1
	for dp in DP:
		test_acc_mean, mem_adv_mean, attr_adv_mean, attack_adv_mean = [], [], [], []
		test_acc_std, mem_adv_std, attr_adv_std, attack_adv_std = [], [], [], []
		for eps in EPSILONS:
			test_acc_d, mem_adv_d, attr_adv_d, attack_adv_d = [], [], [], []
			for run in RUNS:
				train_acc, test_acc, train_loss, membership, attack_adv, attack_pred, mem_adv, mem_pred, attr_adv, attr_mem, attr_pred, features = result[dp][eps][run]
				test_acc_d.append(test_acc)
				mem_adv_d.append(mem_adv) # adversary's advantage using membership inference attack of Yeom et al.
				attack_adv_d.append(attack_adv) # adversary's advantage using membership inference attack of Shokri et al.
				attr_adv_d.append(np.mean(attr_adv)) # adversary's advantage using attribute inference attack of Yeom et al.
			test_acc_mean.append(np.mean(test_acc_d))
			test_acc_std.append(np.std(test_acc_d))
			mem_adv_mean.append(np.mean(mem_adv_d))
			mem_adv_std.append(np.std(mem_adv_d))
			attack_adv_mean.append(np.mean(attack_adv_d))
			attack_adv_std.append(np.std(attack_adv_d))
			attr_adv_mean.append(np.mean(attr_adv_d))
			attr_adv_std.append(np.std(attr_adv_d))
			#print(dp, eps, (baseline_acc - np.mean(test_acc_d)) / baseline_acc, np.std(test_acc_d))
			#print(dp, eps, np.mean(attr_adv_d), np.std(attr_adv_d))
		#plt.errorbar(EPSILONS, (baseline_acc - test_acc_mean) / baseline_acc, yerr=test_acc_std, color=str(color), fmt='.-', capsize=2, label=DP_LABELS[DP.index(dp)])
		plt.errorbar(EPSILONS, attack_adv_mean, yerr=attack_adv_std, color=str(color), fmt='.-', capsize=2, label=DP_LABELS[DP.index(dp)])
		color += 0.2
	
	bottom, top = plt.ylim()
	plt.errorbar(EPS, theoretical_limit(EPS), color='black', fmt='--', capsize=2, label='Theoretical Limit')
	plt.ylim(bottom, top) 
	plt.text(0.2, 0.9, "$\epsilon$-DP Theoretical Limit", color='black', fontsize=12, rotation=80)
	plt.yticks(np.arange(0, 1.1, step=0.2))
	
	plt.xscale('log')
	plt.xlabel('Privacy Budget ($\epsilon$)', fontsize=12)
	#plt.ylabel('Accuracy Loss', fontsize=12)
	#plt.yticks(np.arange(0, 1.1, step=0.2))
	plt.ylabel('Privacy Leakage', fontsize=12)
	#plt.legend()

	plt.text(2, 0.05, "RDP", color='0.7', fontsize=12)
	plt.text(12, 0.055, "zCDP", color='0.5', fontsize=12)
	plt.text(55, 0.7, "Advanced Composition", color='0.3', fontsize=12, rotation=68)
	plt.text(15, -0.05, "Naive Composition", color='0.1', fontsize=12)

	plt.show()


def plot_members_revealed(result):
	thres = 0.05# 0.01 == 1% FPR, 0.02 == 2% FPR, 0.05 == 5% FPR
	_, _, train_loss, membership, _, attack_pred, _, mem_pred, _, attr_mem, attr_pred, _ = pickle.load(open(DATA_PATH+MODEL+'no_privacy_1e-5'+'.p', 'rb'))
	pred = (max(mem_pred) - mem_pred) / (max(mem_pred) - min(mem_pred))
	#pred = attack_pred[:,1]
	print(len(members_revealed(membership, pred, thres)))
	for dp in DP:
		for eps in EPSILONS:
			mems_revealed = []
			for run in RUNS:
				_, _, train_loss, membership, _, attack_pred, _, mem_pred, _, attr_mem, attr_pred, _ = result[dp][eps][run]
				pred = (max(mem_pred) - mem_pred) / (max(mem_pred) - min(mem_pred))
				#pred = attack_pred[:,1]
				mems_revealed.append(members_revealed(membership, pred, thres))
			s = set.intersection(*mems_revealed)
			print(dp, eps, len(s))


def members_revealed(membership, prediction, acceptable_fpr):
	fpr, tpr, thresholds = roc_curve(membership, prediction, pos_label=1)
	l = list(filter(lambda x: x < acceptable_fpr, fpr))
	if len(l) == 0:
		print("Error: low acceptable fpr")
		return None
	threshold = thresholds[len(l)-1]
	#threshold = 0.9
	#print(threshold)
	preds = list(map(lambda val: 1 if val >= threshold else 0, prediction))
	tp = [a*b for a,b in zip(preds,membership)]
	#print(sum(preds) - sum(tp), sum(tp))
	revealed = list(map(lambda i: i if tp[i] == 1 else None, range(len(tp))))
	return set(list(filter(lambda x: x != None, revealed)))
	

def plot_roc(result):
	_, _, train_loss, membership, _, attack_pred, _, mem_pred, _, attr_mem, attr_pred, _ = pickle.load(open(DATA_PATH+MODEL+'no_privacy_1e-8'+'.p', 'rb'))
	pred = attack_pred[:,1]
	val = .9
	fpr, tpr, thresholds = roc_curve(membership, pred, pos_label=1)
	plt.plot(fpr, tpr, color=str(val), label='No Privacy')
	for dp in [DP[0]]:
		for eps in EPSILONS:
			val -= .05
			for run in [1]:
				_, _, train_loss, membership, _, attack_pred, _, mem_pred, _, attr_mem, attr_pred, _ = result[dp][eps][run]
				pred = (max(mem_pred) - mem_pred) / (max(mem_pred) - min(mem_pred))
				#pred = attack_pred[:,1]
				fpr, tpr, thresholds = roc_curve(membership, pred, pos_label=1)
				plt.plot(fpr, tpr, color=str(val), label=eps)
	plt.legend()
	plt.show()
	

result = get_data()
plot_advantage(result) # plot the utility and privacy loss graphs
#plot_members_revealed(result) # return the number of members revealed for different FPR rates
#plot_roc(result)
