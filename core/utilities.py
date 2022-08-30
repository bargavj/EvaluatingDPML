import random
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm

# To avoid numerical inconsistency in calculating log
SMALL_VALUE = 1e-6


def pretty_print_result(mem, pred):
    tn, fp, fn, tp = confusion_matrix(mem, pred).ravel()
    print('TP: %d     FP: %d     FN: %d     TN: %d' % (tp, fp, fn, tn))
    if tp == fp == 0:
        print('PPV: 0\nAdvantage: 0')
    else:
        print('PPV: %.4f\nAdvantage: %.4f' % (tp / (tp + fp), tp / (tp + fn) - fp / (tn + fp)))


def pretty_print_confusion_matrix(cm, labels):
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


def pretty_position(X, Y, pos):
	return ((X[pos] + X[pos+1]) / 2, (Y[pos] + Y[pos+1]) / 2)


def get_ppvs(ref, vec):
    ppv = np.zeros(len(vec))
    vec = sorted(zip(ref, vec), reverse=True, key=(lambda x: x[1]))
    
    vec = [x[0] for x in vec]
    s = 0
    for i, val in enumerate(vec):
        s += val
        if s == 0:
            continue
        ppv[i] = s/(i+1)
    return ppv


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
	return [-np.log(max(b[i,int(a[i])], SMALL_VALUE)) for i in range(len(a))]


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


'''
Decision Tree best parameters seem to vary for different attributes:
Texas Race (4)  - Depth 5, Sample Split 10, Sample Leaf 5
Texas Ethnicity - Depth 4, Sample Split 10, Sample Leaf 5
Census Sex      - Depth 4, Sample Split 10, Sample Leaf 5
Census Race (3) - Depth 5, Sample Split 50, Sample Leaf 25
'''
def fit_model(args, y, x1, x2):
    #clf = svm.SVC(kernel='rbf', gamma=0.7, C=1.0, probability=True)
    if args.dataset == 'census':
        if args.attribute == 1:
            clf = DecisionTreeClassifier(max_depth=5, min_samples_split=50, min_samples_leaf=25)
        else:
            clf = DecisionTreeClassifier(max_depth=4, min_samples_split=10, min_samples_leaf=5)
    else:
        if args.attribute == 1:
            clf = DecisionTreeClassifier(max_depth=5, min_samples_split=10, min_samples_leaf=5)
        else:
            clf = DecisionTreeClassifier(max_depth=4, min_samples_split=10, min_samples_leaf=5)
    if args.comb_flag == 0:
        clf.fit(np.vstack((x1, x2)).T, y)
    else:
        clf.fit(np.vstack((x1, x2, x1 * x2)).T, y)
    return clf


def imputation_training(args, train_x, train_y, test_x, test_y, clf_type='nn', epochs=None):
    if epochs == None:
        epochs = args.target_epochs
    if clf_type == 'nn':
        clf = MLPClassifier(
            random_state=1, 
            max_iter=epochs, 
            hidden_layer_sizes=(args.target_n_hidden, args.target_n_hidden), 
            alpha=args.target_l2_ratio, 
            batch_size=args.target_batch_size, 
            learning_rate_init=args.target_learning_rate)
    else:
        clf = LogisticRegression(
            random_state=1, 
            max_iter=epochs, 
            C=1/args.target_l2_ratio)
    clf.fit(train_x, train_y)
    train_conf = clf.predict_proba(train_x)
    test_conf = clf.predict_proba(test_x)
    train_loss = np.mean(log_loss(train_y, train_conf))
    test_loss = np.mean(log_loss(test_y, test_conf))
    train_acc = clf.score(train_x, train_y)
    test_acc = clf.score(test_x, test_y)
    return test_conf, (train_loss, train_acc, test_loss, test_acc)


def make_line_plot(X, Y, fmt, color, label, lpos=None, loffset=0.02, fsize=18):
    plt.plot(X, np.mean(Y, axis=0), fmt, color=color, label=label)
    plt.fill_between(
        list(range(1, len(X) + 1)),
        np.mean(Y, axis=0) - np.std(Y, axis=0),
        np.mean(Y, axis=0) + np.std(Y, axis=0),
        color=color,
        alpha=0.1)
    if lpos != None:
        plt.text(X[lpos]+np.log10(X[lpos]+0.1), np.mean(Y, axis=0)[lpos]+loffset, label, c=color, fontsize=fsize)


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


def plot_neuron_outputs(plot_info, sorted_neurons, sorted_corr_vals, pos_label, neg_label):
    pos_mean, pos_std, neg_mean, neg_std = plot_info
    x = list(range(len(sorted_neurons)))
    fig, ax1 = plt.subplots()
    ax1.plot(x, neg_mean, '#DAA520', label=neg_label, lw=1)
    ax1.fill_between(x, neg_mean - neg_std, neg_mean + neg_std, alpha=0.2, edgecolor='#DAA520', facecolor='#DAA520')
    ax1.plot(x, pos_mean, '#DC143C', label=pos_label, lw=1)
    ax1.fill_between(x, pos_mean - pos_std, pos_mean + pos_std, alpha=0.2, edgecolor='#DC143C', facecolor='#DC143C')
    ax2 = ax1.twinx()
    plt.plot(x, sorted_corr_vals, 'k', label='correlation', lw=1)
    ax1.text(x[2], neg_mean[2] + 0.07, neg_label, c='#DAA520', fontsize=18)
    ax1.text(x[8], pos_mean[8] + 0.04, pos_label, c='#DC143C', fontsize=18)
    ax2.text(x[2], sorted_corr_vals[2] - 0.14, 'Pearson Correlation', c='k', fontsize=18)
    ax1.set_ylim(0, 1)
    ax2.set_ylim(0, 1)
    ax1.set_xlim(1, len(sorted_neurons))
    ax1.set_xscale('log')
    ax1.set_xlabel('Neuron')
    ax1.set_ylabel('Scaled Neuron Output')
    ax2.set_ylabel('Correlation Value')
    fig.tight_layout()
    plt.show()


def plot_regions(plot_info, skew_label, sensitive_label, dataset='census'):
    frame = plt.rcParams['figure.figsize']
    plt.rcParams['figure.figsize'] = [1.5*frame[0], frame[1]]#[8, 4]
    if dataset == 'census':
        money_label = 'Average Income'
        region_type = 'PUMA'
    else:
        money_label = 'Average Charges'
        region_type = 'Hospital'
    region = [item[0] for item in plot_info]
    money = [item[1] for item in plot_info]
    skew_count = [item[2] for item in plot_info]
    sensitive_count = [item[3] for item in plot_info]
    total_count = [item[4] for item in plot_info]
    fig, ax1 = plt.subplots()
    ax2 = ax1.twinx()
    ax1.plot(range(len(region)), sensitive_count, 'b', alpha=0.5, label=sensitive_label)
    if sensitive_label != skew_label:
        ax1.plot(range(len(region)), skew_count, alpha=0.5, label=skew_label)
    ax1.plot(range(len(region)), total_count, 'orange', alpha=0.5, label='Population')
    ax2.plot(range(len(region)), np.array(money)/1000, 'r', alpha=0.5, label=money_label)
    ax1.set_xlabel('Region (' + region_type + ')')
    ax1.set_ylabel('Number of Records')
    ax2.set_ylabel(money_label + '(X $10^3$)')
    fig.tight_layout()
    ax1.legend(loc='upper right')
    ax2.legend(loc='upper left')
    plt.show()
    
