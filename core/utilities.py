import numpy as np
import random
import matplotlib.pyplot as plt

from core.constants import SMALL_VALUE
from core.constants import SEED
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from collections import Counter


def prety_print_result(mem, pred):
    tn, fp, fn, tp = confusion_matrix(mem, pred).ravel()
    print('TP: %d     FP: %d     FN: %d     TN: %d' % (tp, fp, fn, tn))
    if tp == fp == 0:
    	print('PPV: 0\nAdvantage: 0')
    else:
    	print('PPV: %.4f\nAdvantage: %.4f' % (tp / (tp + fp), tp / (tp + fn) - fp / (tn + fp)))


def prety_print_confusion_matrix(cm, labels):
    cm = np.matrix(cm)
    N = len(labels)
    matrix = [['' for i in range(N+6)] for j in range(N+1)]
    for i in range(N):
        matrix[0][i+1] = matrix[i+1][0] = labels[i][:5]
    matrix[0][N+1] = 'Total'
    matrix[0][N+2] = 'TPR'
    matrix[0][N+3] = 'FPR'
    matrix[0][N+4] = 'Adv'
    matrix[0][N+5] = 'PPV'
    for i in range(N):
        matrix[i+1][N+1] = np.sum(cm[i])
        matrix[i+1][N+2] = cm[i, i] / np.sum(cm[i])
        matrix[i+1][N+3] = (np.sum(cm[:, i]) - cm[i, i]) / (np.sum(cm) - np.sum(cm[i]))
        matrix[i+1][N+4] = str(matrix[i+1][N+2] - matrix[i+1][N+3])[:5]
        matrix[i+1][N+5] = 0 if np.sum(cm[:, i]) == 0 else str(cm[i, i] / np.sum(cm[:, i]))[:5]
        matrix[i+1][N+2] = str(matrix[i+1][N+2])[:5]
        matrix[i+1][N+3] = str(matrix[i+1][N+3])[:5]
        for j in range(N):
            matrix[i+1][j+1] = cm[i,j]
    print('\n'.join([''.join(['{:>8}'.format(val) for val in row]) for row in matrix]))
    print('Accuracy: %.3f' % (np.sum([cm[i, i] for i in range(N)]) / np.sum(cm)))


def get_ppv(mem, pred):
    tn, fp, fn, tp = confusion_matrix(mem, pred).ravel()
    if tp == fp == 0:
    	return 0
    return tp / (tp + fp)


def get_adv(mem, pred):
    tn, fp, fn, tp = confusion_matrix(mem, pred).ravel()
    return (tp / (tp + fn)) - (fp / (tn + fp))


def get_fp(mem, pred):
    tn, fp, fn, tp = confusion_matrix(mem, pred).ravel()
    return fp


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


def log_loss(a, b):
	return [-np.log(max(b[i,a[i]], SMALL_VALUE)) for i in range(len(a))]


def get_random_features(data, pool, size):
    random.seed(SEED)
    features = set()
    while(len(features) < size):
        feat = random.choice(pool)
        c = Counter(data[:,feat])
        # for binary features, select the ones with 1 being minority
        if sorted(list(c.keys())) == [0, 1]:
            if c[1]/len(data) > 0.1 and c[1]/len(data) < 0.5:
                features.add(feat)
        # select feature that has more than one value
        elif len(c.keys()) > 1:
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
            mem[int(signs[i] * trials)] += 1
        else:
            non_mem[int(signs[i] * trials)] += 1
    plt.plot(np.arange(0, 1.01, 0.01), mem / mem_size, 'k-', label='Members')
    plt.plot(np.arange(0, 1.01, 0.01), non_mem / non_mem_size, 'k--', label='Non Members')
    plt.xlabel('Merlin Ratio')
    plt.ylabel('Fraction of Instances')
    plt.xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    plt.yticks(np.arange(0, 0.11, step=0.02))
    plt.xlim(0, 1.0)
    plt.ylim(0, 0.1)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_histogram(vector):
    mem = vector[:10000]
    non_mem = vector[10000:]
    data, bins, _ = plt.hist([mem, non_mem], bins=loss_range())
    plt.clf()
    mem_hist = np.array(data[0])
    non_mem_hist = np.array(data[1])
    plt.plot(bins[:-1], mem_hist / len(mem), 'k-', label='Members')
    plt.plot(bins[:-1], non_mem_hist / len(non_mem), 'k--', label='Non Members')
    plt.xscale('log')
    plt.xticks([10**-6, 10**-4, 10**-2, 10**0])
    plt.yticks(np.arange(0, 0.11, step=0.02))
    plt.ylim(0, 0.1)
    plt.xlabel('Per-Instance Loss')
    plt.ylabel('Fraction of Instances')
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_ai_histogram(vector, positive_indices, positive_label, lab):
    #fig = plt.figure()
    pos = vector[positive_indices]
    neg = vector[list(set(range(len(vector))) - set(positive_indices))]
    data, bins, _ = plt.hist([pos, neg], bins=np.arange(0, 1.01, step=0.01))
    plt.clf()
    pos_hist = np.array(data[0])
    neg_hist = np.array(data[1])
    plt.plot(bins[:-1], pos_hist / len(pos), '-m', label=positive_label)
    plt.plot(bins[:-1], neg_hist / len(neg), '-y', label='Not '+ positive_label)
    plt.xticks(np.arange(0, 1.1, step=0.2))
    plt.yticks(np.arange(0, 0.11, step=0.02))
    plt.ylim(0, 0.1)
    plt.xlabel('Probability of Predicting ' + positive_label)
    plt.ylabel('Fraction of Instances')
    plt.legend()
    plt.tight_layout()
    plt.show()
    #fig.savefig(str(lab) + ".pdf", format='pdf', dpi=1000, bbox_inches='tight')


def plot_layer_outputs(plot_info, pos_label, neg_label, informative_neurons):
    pos_mean, pos_std, neg_mean, neg_std = plot_info
    x = list(range(len(informative_neurons)))
    plt.plot(x, neg_mean, '#DAA520', label=neg_label, lw=1)
    plt.fill_between(x, neg_mean - neg_std, neg_mean + neg_std, alpha=0.2, edgecolor='#DAA520', facecolor='#DAA520')
    plt.plot(x, pos_mean, '#DC143C', label=pos_label, lw=1)
    plt.fill_between(x, pos_mean - pos_std, pos_mean + pos_std, alpha=0.2, edgecolor='#DC143C', facecolor='#DC143C')
    plt.xticks(ticks = x, labels = [str(val) for val in informative_neurons])
    plt.xlabel('Neuron Number')
    plt.ylabel('Neuron Output')
    plt.legend()
    plt.tight_layout()
    plt.show()
