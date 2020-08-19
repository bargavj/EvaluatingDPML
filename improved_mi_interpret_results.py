from sklearn.metrics import roc_curve
from utilities import get_fp, get_adv, get_ppv, get_inference_threshold, plot_histogram, plot_sign_histogram
import matplotlib.pyplot as plt
import numpy as np
import pickle
import argparse

EPS = list(np.arange(0.1, 100, 0.01))
EPS2 = list(np.arange(0.1, 100, 0.01))
EPSILONS = [0.1, 1.0, 10.0, 100.0]
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

def get_pred_mem_mi(per_instance_loss, proposed_mi_outputs, method='yeom', fpr_threshold=None, per_class_thresh=False, fixed_thresh=False):
	# method == "yeom" runs an improved version of the Yeom attack that finds a better threshold than the original
	# method == "merlin" runs a new attack, which uses the direction of the change in per-instance loss for the record
	true_y, v_true_y, v_membership, v_per_instance_loss, v_counts, counts = proposed_mi_outputs
	if method == 'yeom':
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
	else:  # In this case, run the Merlin attack.
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

def plot_distributions(pred_vector, true_vector, method='yeom'):
	fpr, tpr, phi = roc_curve(true_vector, pred_vector, pos_label=1)
	fpr, tpr, phi = np.array(fpr), np.array(tpr), np.array(phi)
	if method == 'yeom':
		fpr = 1 - fpr
		tpr = 1 - tpr
	PPV_A = tpr / (tpr + gamma * fpr)
	Adv_A = tpr - fpr
	fig, ax1 = plt.subplots()
	if method == 'yeom':
		phi, fpr, Adv_A, PPV_A = phi[:-1], fpr[:-1], Adv_A[:-1], PPV_A[:-1]
	ax1.plot(phi, Adv_A, label="Adv", color='green')
	ax1.plot(phi, PPV_A, label="PPV", color='orange')
	ax2 = ax1.twinx()
	ax2.plot(phi, fpr, label="FPR", color='black', linestyle='dashed')
	if method == 'yeom':
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

def get_zeros(mem, vect):
	mem_zeros = list(filter(lambda mem[i]: vect[i] == 0, list(range(len(mem)))))
	#print(np.mean(vect[:10000]), np.std(vect[:10000]))
	#print(np.mean(vect[10000:]), np.std(vect[10000:]))
	return np.sum(mem_zeros), len(mem_zeros) - np.sum(mem_zeros) 

def plot_accuracy(result):
	train_accs, baseline_acc = np.zeros(B), np.zeros(B)
	for run in RUNS:
		aux, membership, per_instance_loss, yeom_mi_outputs_1, yeom_mi_outputs_2, proposed_mi_outputs = result['no_privacy'][run]
		train_loss, train_acc, test_loss, test_acc = aux
		baseline_acc[run] = test_acc
		train_accs[run] = train_acc				
	baseline_acc = np.mean(baseline_acc)
	print(np.mean(train_accs), baseline_acc)
	color = 0.1
	y = dict()
	for dp in DP:
		test_acc_vec = np.zeros((A, B))
		for a, eps in enumerate(EPSILONS):
			for run in RUNS:
				aux, membership, per_instance_loss, yeom_mi_outputs_1, yeom_mi_outputs_2, proposed_mi_outputs = result[dp][eps][run]
				train_loss, train_acc, test_loss, test_acc = aux
				test_acc_vec[a, run] = test_acc			
		y[dp] = 1 - np.mean(test_acc_vec, axis=1) / baseline_acc
		plt.errorbar(EPSILONS, y[dp], yerr=np.std(test_acc_vec, axis=1), color=str(color), fmt='.-', capsize=2, label=DP_LABELS[DP.index(dp)])
		color += 0.2
	plt.xscale('log')
	plt.xlabel('Privacy Budget ($\epsilon$)')	
	plt.ylabel('Accuracy Loss')
	plt.yticks(np.arange(0, 1.1, step=0.2))
	plt.annotate("RDP", pretty_position(EPSILONS, y["rdp_"], 2), textcoords="offset points", xytext=(20, 10), ha='right', color=str(0.3))
	plt.annotate("GDP", pretty_position(EPSILONS, y["gdp_"], 2), textcoords="offset points", xytext=(-20, -10), ha='right', color=str(0.1))
	plt.tight_layout()
	plt.show()

def plot_privacy_leakage(result, eps=None, dp='gdp_'):
	adv_yeom_vanilla_1, adv_yeom, adv_merlin = np.zeros(B), np.zeros(B), np.zeros(B)
	ppv_yeom_vanilla_1, ppv_yeom, ppv_merlin = np.zeros(B), np.zeros(B), np.zeros(B)
	fpr_yeom_vanilla_1, fpr_yeom, fpr_merlin = np.zeros(B), np.zeros(B), np.zeros(B)
	thresh_yeom_vanilla_1, thresh_yeom, thresh_merlin = np.zeros(B), np.zeros(B), np.zeros(B)
	yeom_zero_m, yeom_zero_nm, merlin_zero_m, merlin_zero_nm = [], [], [], []
	for run in RUNS:
		aux, membership, per_instance_loss, yeom_mi_outputs_1, yeom_mi_outputs_2, proposed_mi_outputs = result['no_privacy'][run] if not eps else result[dp][eps][run]
		train_loss, train_acc, test_loss, test_acc = aux
		true_y, v_true_y, v_membership, v_per_instance_loss, v_counts, counts = proposed_mi_outputs
		m, nm = get_zeros(membership, per_instance_loss)
		yeom_zero_m.append(m)
		yeom_zero_nm.append(nm)
		m, nm = get_zeros(membership, counts)
		merlin_zero_m.append(m)
		merlin_zero_nm.append(nm)
		#plot_histogram(per_instance_loss)
		plot_distributions(per_instance_loss, membership, method='yeom')
		#plot_sign_histogram(membership, counts, 100)
		plot_distributions(counts, membership, method='merlin')
		# As used below, method == 'yeom' runs a Yeom attack but finds a better threshold than is used in the original Yeom attack.
		thresh, pred = get_pred_mem_mi(per_instance_loss, proposed_mi_outputs, method='yeom', fpr_threshold=alpha, per_class_thresh=args.per_class_thresh, fixed_thresh=args.fixed_thresh)
		fp, adv, ppv = get_fp(membership, pred), get_adv(membership, pred), get_ppv(membership, pred)
		thresh_yeom[run], fpr_yeom[run], adv_yeom[run], ppv_yeom[run] = thresh, fp / (gamma * 10000), adv, ppv
		# As used below, method == 'merlin' runs a new threshold-based membership inference attack that uses the direction of the change in per-instance loss for the record.
		thresh, pred = get_pred_mem_mi(per_instance_loss, proposed_mi_outputs, method='merlin', fpr_threshold=alpha, per_class_thresh=args.per_class_thresh, fixed_thresh=args.fixed_thresh)
		fp, adv, ppv = get_fp(membership, pred), get_adv(membership, pred), get_ppv(membership, pred)
		thresh_merlin[run], fpr_merlin[run], adv_merlin[run], ppv_merlin[run] = thresh, fp / (gamma * 10000), adv, ppv
		# Original Yeom attack that uses expected training loss threshold
		fp, adv, ppv = get_fp(membership, yeom_mi_outputs_1), get_adv(membership, yeom_mi_outputs_1), get_ppv(membership, yeom_mi_outputs_1)
		thresh_yeom_vanilla_1[run], fpr_yeom_vanilla_1[run], adv_yeom_vanilla_1[run], ppv_yeom_vanilla_1[run] = train_loss, fp / (gamma * 10000), adv, ppv
	print('\nYeom: \t %.2f +/- %.2f \t %.2f +/- %.2f' % (np.mean(yeom_zero_m), np.std(yeom_zero_m), np.mean(yeom_zero_nm), np.std(yeom_zero_nm)))
	print('\nMerlin: \t %.2f +/- %.2f \t %.2f +/- %.2f' % (np.mean(merlin_zero_m), np.std(merlin_zero_m), np.mean(merlin_zero_nm), np.std(merlin_zero_nm)))
	print('\nYeom Vanilla 1:\nphi: %f +/- %f\nFPR: %.4f +/- %.4f\nTPR: %.4f +/- %.4f\nAdv: %.4f +/- %.4f\nPPV: %.4f +/- %.4f' % (np.mean(thresh_yeom_vanilla_1), np.std(thresh_yeom_vanilla_1), np.mean(fpr_yeom_vanilla_1), np.std(fpr_yeom_vanilla_1), np.mean(adv_yeom_vanilla_1+fpr_yeom_vanilla_1), np.std(adv_yeom_vanilla_1+fpr_yeom_vanilla_1), np.mean(adv_yeom_vanilla_1), np.std(adv_yeom_vanilla_1), np.mean(ppv_yeom_vanilla_1), np.std(ppv_yeom_vanilla_1)))
	print('\nYeom:\nphi: %f +/- %f\nFPR: %.4f +/- %.4f\nTPR: %.4f +/- %.4f\nAdv: %.4f +/- %.4f\nPPV: %.4f +/- %.4f' % (np.mean(thresh_yeom), np.std(thresh_yeom), np.mean(fpr_yeom), np.std(fpr_yeom), np.mean(adv_yeom+fpr_yeom), np.std(adv_yeom+fpr_yeom), np.mean(adv_yeom), np.std(adv_yeom), np.mean(ppv_yeom), np.std(ppv_yeom)))
	print('\nMerlin:\nphi: %f +/- %f\nFPR: %.4f +/- %.4f\nTPR: %.4f +/- %.4f\nAdv: %.4f +/- %.4f\nPPV: %.4f +/- %.4f' % (np.mean(thresh_merlin), np.std(thresh_merlin), np.mean(fpr_merlin), np.std(fpr_merlin), np.mean(adv_merlin+fpr_merlin), np.std(adv_merlin+fpr_merlin), np.mean(adv_merlin), np.std(adv_merlin), np.mean(ppv_merlin), np.std(ppv_merlin)))				

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('dataset', type=str)
	parser.add_argument('--model', type=str, default='nn')
	parser.add_argument('--l2_ratio', type=str, default='1e-08')
	parser.add_argument('--gamma', type=float, default=1.0)
	parser.add_argument('--alpha', type=float, default=None)
	parser.add_argument('--per_class_thresh', type=int, default=0)
	parser.add_argument('--fixed_thresh', type=int, default=0)
	parser.add_argument('--plot', type=str, default='acc')
	parser.add_argument('--eps', type=float, default=None)
	args = parser.parse_args()
	print(vars(args))

	gamma = args.gamma
	alpha = args.alpha
	DATA_PATH = 'results/' + str(args.dataset) + '_improved_mi/'
	MODEL = str(gamma) + '_' + str(args.model) + '_'

	result = get_data()
	if args.plot == 'acc':
		plot_accuracy(result)
	else:
		plot_privacy_leakage(result, args.eps)
