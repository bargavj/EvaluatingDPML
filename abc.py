from utilities import get_fp, get_adv, get_ppv
from improved_mi_interpret_results import get_pred_mem_mi
import matplotlib.pyplot as plt
import numpy as np
import pickle

DATA_PATH = 'results/purchase_100/'
MODEL = '1_nn_'
alpha = 0.0003
model = 'priv'
attack = 'merlin'
thresh_attack, fpr_attack, adv_attack, ppv_attack = np.zeros((5,5)), np.zeros((5,5)), np.zeros((5,5)), np.zeros((5,5))
vals = []

for run in range(5):
	if model == 'priv':
		vals.append(list(pickle.load(open(DATA_PATH+MODEL+'grad_pert_gdp_100.0_'+str(run+1)+'.p', 'rb'))))
	else:
		vals.append(list(pickle.load(open(DATA_PATH+MODEL+'no_privacy_1e-08_'+str(run+1)+'.p', 'rb'))))

	aux, membership, per_instance_loss, yeom_mi_outputs_1, yeom_mi_outputs_2, proposed_mi_outputs_arr = vals[run]
	train_loss, train_acc, test_loss, test_acc = aux
	for i in range(5):
		thresh, pred = get_pred_mem_mi(per_instance_loss, proposed_mi_outputs_arr[i], method=attack, fpr_threshold=alpha)
		fp, adv, ppv = get_fp(membership, pred), get_adv(membership, pred), get_ppv(membership, pred)
		print(thresh, fp, adv*10000 + fp)
		thresh_attack[run][i], fpr_attack[run][i], adv_attack[run][i], ppv_attack[run][i] = thresh, fp / 10000, adv, ppv
	print('\nRun %d:\nphi: %f +/- %f\nFPR: %.4f +/- %.4f\nTPR: %.4f +/- %.4f\nAdv: %.4f +/- %.4f\nPPV: %.4f +/- %.4f' % (run+1, np.mean(thresh_attack[run]), np.std(thresh_attack[run]), np.mean(fpr_attack[run]), np.std(fpr_attack[run]), np.mean(adv_attack[run]+fpr_attack[run]), np.std(adv_attack[run]+fpr_attack[run]), np.mean(adv_attack[run]), np.std(adv_attack[run]), np.mean(ppv_attack[run]), np.std(ppv_attack[run])))
print('\n\nAcross All Runs:\nphi: %f +/- %f\nFPR: %.4f +/- %.4f\nTPR: %.4f +/- %.4f\nAdv: %.4f +/- %.4f\nPPV: %.4f +/- %.4f\n' % (np.mean(thresh_attack), np.std(thresh_attack), np.mean(fpr_attack), np.std(fpr_attack), np.mean(adv_attack+fpr_attack), np.std(adv_attack+fpr_attack), np.mean(adv_attack), np.std(adv_attack), np.mean(ppv_attack), np.std(ppv_attack)))
