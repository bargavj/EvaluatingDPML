from sklearn.metrics import confusion_matrix, roc_curve
import numpy as np
import random
import matplotlib.pyplot as plt

# to avoid numerical inconsistency in calculating log
SMALL_VALUE = 1e-6

def prety_print_result(mem, pred):
    tn, fp, fn, tp = confusion_matrix(mem, pred).ravel()
    print('TP: %d     FP: %d     FN: %d     TN: %d' % (tp, fp, fn, tn))
    if tp == fp == 0:
    	print('PPV: 0\nAdvantage: 0')
    else:
    	print('PPV: %.4f\nAdvantage: %.4f' % (tp / (tp + fp), tp / (tp + fn) - fp / (tn + fp)))

def get_ppv(mem, pred):
    tn, fp, fn, tp = confusion_matrix(mem, pred).ravel()
    if tp == fp == 0:
    	return 0
    return tp / (tp + fp)

def get_fp_adv_ppv(mem, pred):
    tn, fp, fn, tp = confusion_matrix(mem, pred).ravel()
    if tp == fp == 0:
    	return 0, 0, 0
    return fp, (tp / (tp + fn)) - (fp / (tn + fp)), tp / (tp + fp)

def get_inference_threshold(pred_vector, true_vector, fpr_threshold=None):
    fpr, tpr, thresholds = roc_curve(true_vector, pred_vector, pos_label=1)
    # return inference threshold corresponding to maximum advantage
    if fpr_threshold == None:
    	return thresholds[np.argmax(tpr-fpr)]
    # return inference threshold corresponding to fpr_threshold
    for a, b in zip(fpr, thresholds):
    	if a > fpr_threshold:
    		break
    	alpha_thresh = b
    return alpha_thresh

def loss_range():
	return [10**i for i in np.arange(-7, 1, 0.1)]
	#return list(np.arange(0, -np.log(SMALL_VALUE), 0.0001))

def log_loss(a, b):
	return [-np.log(max(b[i,a[i]], SMALL_VALUE)) for i in range(len(a))]

def get_random_features(data, pool, size):
    random.seed(21312)
    features = set()
    while(len(features) < size):
        feat = random.choice(pool)
        if len(np.unique(data[:,feat])) > 1:
            features.add(feat)
    return list(features)

def get_attribute_variations(data, feature):
	if len(np.unique(data[:,feature])) == 2:
		low, high = np.unique(data[:,feature])
		pivot = (low + high) / 2
	else:
		pivot = np.quantile(np.unique(data[:,feature]), 0.5)
		low = np.quantile(np.unique(data[:,feature]), 0.25)
		high = np.quantile(np.unique(data[:,feature]), 0.75)
	true_attribute_value = np.where(data[:,feature] <= pivot, 0, 1)
	return low, high, true_attribute_value

def generate_noise(shape, dtype, noise_params):
    noise_type, noise_coverage, noise_magnitude = noise_params
    if noise_coverage == 'full':
        if noise_type == 'uniform':
            return np.array(np.random.uniform(0, noise_magnitude, size=shape), dtype=dtype)
        else:
            return np.array(np.random.normal(0, noise_magnitude, size=shape), dtype=dtype)
    attr = np.random.randint(shape[1])
    noise = np.zeros(shape, dtype=dtype)
    if noise_type == 'uniform':
        noise[:, attr] = np.array(np.random.uniform(0, noise_magnitude, size=shape[0]), dtype=dtype)
    else:
        noise[:, attr] = np.array(np.random.normal(0, noise_magnitude, size=shape[0]), dtype=dtype)
    return noise

def plot_sign_histogram(membership, signs, trials):
    signs = np.array(signs, dtype='int32')
    mem, non_mem = np.zeros(trials + 1), np.zeros(trials + 1)
    mem_size, non_mem_size = sum(membership), len(membership) - sum(membership)
    for i in range(len(signs)):
        if membership[i] == 1:
            mem[signs[i]] += 1
        else:
            non_mem[signs[i]] += 1
    plt.plot(np.arange(trials + 1), mem / mem_size, 'k-', label='Members')
    plt.plot(np.arange(trials + 1), non_mem / non_mem_size, 'k--', label='Non Members')
    plt.xlabel('Number of Times Loss Increases (out of '+str(trials)+')')
    plt.ylabel('Fraction of Instances')
    plt.xticks(list(range(0, trials + 1, trials // 5)))
    plt.yticks(np.arange(0, 0.11, step=0.02))
    plt.ylim(0, 0.1)
    plt.legend()
    plt.tight_layout()
    plt.show()

def plot_histogram(vector):
    mem = vector[:10000]
    non_mem = vector[10000:]
    #true_vector = np.concatenate((np.ones(10000, dtype='int32'), np.zeros(len(vector) - 10000, dtype='int32')))
    #fpr, tpr, phi = roc_curve(true_vector, vector, pos_label=1)
    #data, bins, _ = plt.hist([mem, non_mem], bins=list(reversed(phi)))
    data, bins, _ = plt.hist([mem, non_mem], bins=loss_range())
    plt.clf()
    mem_hist = np.array(data[0])
    non_mem_hist = np.array(data[1])
    plt.plot(bins[:-1], mem_hist / len(mem), 'k-', label='Members')
    plt.plot(bins[:-1], non_mem_hist / len(non_mem), 'k--', label='Non Members')
    plt.xscale('log')
    #plt.yscale('log')
    plt.xticks([10**-6, 10**-4, 10**-2, 10**0])
    plt.yticks(np.arange(0, 0.11, step=0.02))
    plt.ylim(0, 0.1)
    plt.xlabel('Per-Instance Loss')
    plt.ylabel('Fraction of Instances')
    plt.legend()
    plt.tight_layout()
    plt.show()

def make_membership_box_plot(vector):
    plt.boxplot([vector[:10000], vector[10000:]], labels=['members', 'non-members'], whis='range')
    plt.yscale('log')
    plt.ylabel('Per-Instance Loss')
    plt.show()

def make_predictions_box_plot(vector, mem, pred_mem):
    tp_vec = [vector[i] for i in range(len(vector)) if mem[i] == 1 and pred_mem[i] == 1]
    fn_vec = [vector[i] for i in range(len(vector)) if mem[i] == 1 and pred_mem[i] == 0]
    fp_vec = [vector[i] for i in range(len(vector)) if mem[i] == 0 and pred_mem[i] == 1]
    tn_vec = [vector[i] for i in range(len(vector)) if mem[i] == 0 and pred_mem[i] == 0]
    plt.boxplot([tp_vec, fn_vec, fp_vec, tn_vec], labels=['TP', 'FN', 'FP', 'TN'], whis='range')
    plt.yscale('log')
    plt.ylabel('Per-Instance Loss')
    plt.show()