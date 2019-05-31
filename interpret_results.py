from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse


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
	train_acc, baseline_acc, train_loss, membership, _, attack_pred, _, mem_pred, _, attr_mem, attr_pred, _ = pickle.load(open(DATA_PATH+MODEL+'no_privacy_'+str(args.l2_ratio)+'.p', 'rb'))
	print(train_acc, baseline_acc)
	color = 0.1
	y = dict()
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

			if args.silent == 0:
				if args.plot == 'acc':
					print(dp, eps, (baseline_acc - np.mean(test_acc_d)) / baseline_acc, np.std(test_acc_d))
				elif args.plot == 'attack':
					print(dp, eps, np.mean(attack_adv_d), np.std(attack_adv_d))
				elif args.plot == 'attr':
					print(dp, eps, np.mean(attr_adv_d), np.std(attr_adv_d))
				elif args.plot == 'mem':
					print(dp, eps, np.mean(mem_adv_d), np.std(mem_adv_d))
		if args.plot == 'acc':
			y[dp] = (baseline_acc - test_acc_mean) / baseline_acc
			plt.errorbar(EPSILONS, (baseline_acc - test_acc_mean) / baseline_acc, yerr=test_acc_std, color=str(color), fmt='.-', capsize=2, label=DP_LABELS[DP.index(dp)])
		elif args.plot == 'attack':
			y[dp] = attack_adv_mean
			plt.errorbar(EPSILONS, attack_adv_mean, yerr=attack_adv_std, color=str(color), fmt='.-', capsize=2, label=DP_LABELS[DP.index(dp)])
		elif args.plot == 'attr':
			y[dp] = attr_adv_mean
			plt.errorbar(EPSILONS, attr_adv_mean, yerr=attr_adv_std, color=str(color), fmt='.-', capsize=2, label=DP_LABELS[DP.index(dp)])
		elif args.plot == 'mem':
			y[dp] = mem_adv_mean
			plt.errorbar(EPSILONS, mem_adv_mean, yerr=mem_adv_std, color=str(color), fmt='.-', capsize=2, label=DP_LABELS[DP.index(dp)])
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


def members_revealed(result):
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

	DATA_PATH = 'results/' + str(args.dataset) + '/'
	MODEL = str(args.model) + '_'

	result = get_data()
	if args.function == 1:
		plot_advantage(result) # plot the utility and privacy loss graphs
	else:
		members_revealed(result) # return the number of members revealed for different FPR rates
