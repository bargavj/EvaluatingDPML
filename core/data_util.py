import os
import pickle
import random
import numpy as np
import pandas as pd

from core.utilities import plot_regions
from sklearn.model_selection import train_test_split
from collections import Counter

# Seed for random number generator
SEED = 21312

def load_attack_data(MODEL_PATH):
    """
    Helper function to load the attack data for meta-model training.
    """
    fname = MODEL_PATH + 'attack_train_data.npz'
    with np.load(fname) as f:
        train_x, train_y = [f['arr_%d' % i] for i in range(len(f.files))]
    fname = MODEL_PATH + 'attack_test_data.npz'
    with np.load(fname) as f:
        test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]
    return train_x.astype('float32'), train_y.astype('int32'), test_x.astype('float32'), test_y.astype('int32')


def save_data(args, data_id=None):
    """
    Function to create the training, test and hold-out sets 
    by randomly sampling from the raw data set.
    """
    print('-' * 10 + 'SAVING DATA TO DISK' + '-' * 10 + '\n')
    target_size = args.target_data_size
    gamma = args.target_test_train_ratio
    DATA_PATH = 'data/' + args.train_dataset + '/'
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    if data_id and int(data_id) >= 0:
        x = pickle.load(open('../dataset/'+args.train_dataset+f'_features_{data_id}.p', 'rb'))
        y = pickle.load(open('../dataset/'+args.train_dataset+f'_labels_{data_id}.p', 'rb'))
    else:
        x = pickle.load(open('../dataset/'+args.train_dataset+'_features.p', 'rb'))
        y = pickle.load(open('../dataset/'+args.train_dataset+'_labels.p', 'rb'))
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(x.shape, y.shape)

    # assert if data is enough for sampling target data
    assert(len(x) >= (1 + gamma) * target_size)
    x, train_x, y, train_y = train_test_split(x, y, test_size=target_size, stratify=y)
    print("Training set size:  X: {}, y: {}".format(train_x.shape, train_y.shape))
    x, test_x, y, test_y = train_test_split(x, y, test_size=int(gamma*target_size), stratify=y)
    print("Test set size:  X: {}, y: {}".format(test_x.shape, test_y.shape))

    # save target data
    print('Saving data for target model')
    if data_id and int(data_id) >= 0:
        np.savez(DATA_PATH + f'target_data_{data_id}.npz', train_x, train_y, test_x, test_y)
    else:
        np.savez(DATA_PATH + 'target_data.npz', train_x, train_y, test_x, test_y)

    # assert if remaining data is enough for sampling shadow data
    assert(len(x) >= (1 + gamma) * target_size)

    # save shadow data
    for i in range(args.n_shadow):
        print('Saving data for shadow model {}'.format(i))
        train_x, test_x, train_y, test_y = train_test_split(x, y, train_size=target_size, test_size=int(gamma*target_size), stratify=y)
        print("Training set size:  X: {}, y: {}".format(train_x.shape, train_y.shape))
        print("Test set size:  X: {}, y: {}".format(test_x.shape, test_y.shape))
        if data_id and data_id >= 0:
            np.savez(DATA_PATH + 'shadow{}_data_{}.npz'.format(i, data_id), train_x, train_y, test_x, test_y)
        else:
            np.savez(DATA_PATH + 'shadow{}_data.npz'.format(i), train_x, train_y, test_x, test_y)

            
def sample_noniid_data(args):
    """
    Function to create the training, test and hold-out sets 
    by custom non-iid sampling from the raw data set.
    """
    # this functionality is currently only for census and texas data set
    assert(args.train_dataset == 'census' or args.train_dataset == 'texas_100_v2')
    
    print('-' * 10 + 'SAMPLING NON-IID DATA' + '-' * 10 + '\n')
    target_size = args.target_data_size
    gamma = args.target_test_train_ratio
    DATA_PATH = 'data/' + args.train_dataset + '/'
    if not os.path.exists(DATA_PATH):
        os.makedirs(DATA_PATH)
    x = pickle.load(open('../dataset/'+args.train_dataset+'_features.p', 'rb'))
    y = pickle.load(open('../dataset/'+args.train_dataset+'_labels.p', 'rb'))
    x = np.array(x, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(x.shape, y.shape)
    
    # assert if data is enough for sampling data sub-sets
    assert(len(x) >= (1 + gamma) * 4 * target_size)
    
    target_attrs, attribute_dict, max_attr_vals, col_flags = get_sensitive_features(args.train_dataset, x)
    if args.train_dataset == 'census':
        assert(col_flags['CENSUS_ST'] != None and col_flags['CENSUS_PUMA'] != None and col_flags['CENSUS_PINCP'] != None)
        # target attribute is Race for Census
        target_attr = target_attrs[1]
        money_attr = col_flags['CENSUS_PINCP']
    elif args.train_dataset == 'texas_100_v2':
        assert(col_flags['TEXAS_THCIC_ID'] != None and col_flags['TEXAS_TOTAL_CHARGES'] != None)
        # target attribute is Ethnicity for Texas
        target_attr = target_attrs[2]
        money_attr = col_flags['TEXAS_TOTAL_CHARGES']
    
    # sample target data from general distribution
    x, train_x, y, train_y = train_test_split(x, y, test_size=target_size, stratify=y)
    print("Target Train set size:  X: {}, y: {}".format(train_x.shape, train_y.shape))
    x, test_x, y, test_y = train_test_split(x, y, test_size=int(gamma*target_size), stratify=y)
    print("Target Test set size:  X: {}, y: {}".format(test_x.shape, test_y.shape))
    print(np.sum(train_x[:, target_attr] * max_attr_vals[target_attr] == args.sensitive_outcome) / target_size, np.sum(test_x[:, target_attr] * max_attr_vals[target_attr] == args.sensitive_outcome) / int(gamma*target_size))
    print(Counter(train_x[:, target_attr] * max_attr_vals[target_attr]), len(Counter(train_y).keys()))
    print('Saving target data\n')
    np.savez(DATA_PATH + 'target_data.npz', train_x, train_y, test_x, test_y)
    
    # sample holdout data from general distribution
    x, h_train_x, y, h_train_y = train_test_split(x, y, test_size=target_size, stratify=y)
    print("Holdout Train set size:  X: {}, y: {}".format(h_train_x.shape, h_train_y.shape))
    x, h_test_x, y, h_test_y = train_test_split(x, y, test_size=int(gamma*target_size), stratify=y)
    print("Holdout Test set size:  X: {}, y: {}".format(h_test_x.shape, h_test_y.shape))
    print(np.sum(h_train_x[:, target_attr] * max_attr_vals[target_attr] == args.sensitive_outcome) / target_size, np.sum(h_test_x[:, target_attr] * max_attr_vals[target_attr] == args.sensitive_outcome) / int(gamma*target_size))
    print(Counter(h_train_x[:, target_attr] * max_attr_vals[target_attr]), len(Counter(h_train_y).keys()))
    print('Saving target holdout data')
    np.savez(DATA_PATH + 'holdout_data.npz', h_train_x, h_train_y, h_test_x, h_test_y)
    
    # group the records into regions
    region = {}
    for idx, record in enumerate(x):
        if args.train_dataset == 'texas_100_v2':
            r = str(record[col_flags['TEXAS_THCIC_ID']] * max_attr_vals[col_flags['TEXAS_THCIC_ID']])
        else:
            r = str(record[col_flags['CENSUS_ST']] * max_attr_vals[col_flags['CENSUS_ST']]) + '+' + str(record[col_flags['CENSUS_PUMA']] * max_attr_vals[col_flags['CENSUS_PUMA']])
        if r not in region:
            region[r] = [idx]
        else:
            region[r].append(idx)
    print(len(region))
    plot_info = [[
        r, 
        np.mean(x[indices, money_attr] * max_attr_vals[money_attr]), 
        np.sum(x[indices, target_attr] * max_attr_vals[target_attr] == args.skew_outcome), 
        np.sum(x[indices, target_attr] * max_attr_vals[target_attr] == args.sensitive_outcome), 
        len(indices)
    ] for r, indices in region.items()]
    # sort regions based on Income or Charges (High to Low)
    if args.skew_attribute == 1:
        plot_info = sorted(plot_info, key=(lambda x: x[1]), reverse=True)
    # sort regions based on Race or Ethnicity (High to Low)
    elif args.skew_attribute == 2:
        plot_info = sorted(plot_info, key=(lambda x: x[2]), reverse=True)
    # sort regions based on Population (High to Low)
    else:
        plot_info = sorted(plot_info, key=(lambda x: x[4]), reverse=True)
    plot_regions(plot_info, attribute_dict[target_attr][args.skew_outcome], attribute_dict[target_attr][args.sensitive_outcome], args.train_dataset)
    
    sk_idx = []
    i = 0
    region_list = []
    while len(sk_idx) < (1 + gamma) * target_size:
        sk_idx.extend(region[plot_info[i][0]])
        region_list.append(i)
        i += 1
    print(len(region_list))
    # sample skewed data from skewed distribution (most populous hospitals)
    sk_train_x, sk_test_x, sk_train_y, sk_test_y = train_test_split(x[sk_idx], y[sk_idx], train_size=target_size, test_size=int(gamma*target_size), stratify=y[sk_idx])
    print("Skewed Train set size:  X: {}, y: {}".format(sk_train_x.shape, sk_train_y.shape))
    print("Skewed Test set size:  X: {}, y: {}".format(sk_test_x.shape, sk_test_y.shape))
    print(np.sum(sk_train_x[:, target_attr] * max_attr_vals[target_attr] == args.sensitive_outcome) / target_size, np.sum(sk_test_x[:, target_attr] * max_attr_vals[target_attr] == args.sensitive_outcome) / int(gamma*target_size))
    print(Counter(sk_train_x[:, target_attr] * max_attr_vals[target_attr]), len(Counter(sk_train_y).keys()))
    print('Saving skewed data\n')
    np.savez(DATA_PATH + 'skewed_data.npz', sk_train_x, sk_train_y, sk_test_x, sk_test_y)
    
    sk_idx = []
    i = len(region) - 1
    region_list = []
    while len(sk_idx) < (1 + gamma) * target_size:
        sk_idx.extend(region[plot_info[i][0]])
        region_list.append(i)
        i -= 1
    print(len(region_list))
    # sample skewed 2 data from skewed distribution (least populous hospitals)
    sk2_train_x, sk2_test_x, sk2_train_y, sk2_test_y = train_test_split(x[sk_idx], y[sk_idx], train_size=target_size, test_size=int(gamma*target_size), stratify=y[sk_idx])
    print("Skewed 2 Train set size:  X: {}, y: {}".format(sk2_train_x.shape, sk2_train_y.shape))
    print("Skewed 2 Test set size:  X: {}, y: {}".format(sk2_test_x.shape, sk2_test_y.shape))
    print(np.sum(sk2_train_x[:, target_attr] * max_attr_vals[target_attr] == args.sensitive_outcome) / target_size, np.sum(sk2_test_x[:, target_attr] * max_attr_vals[target_attr] == args.sensitive_outcome) / int(gamma*target_size))
    print(Counter(sk2_train_x[:, target_attr] * max_attr_vals[target_attr]), len(Counter(sk2_train_y).keys()))
    print('Saving skewed 2 data\n')
    np.savez(DATA_PATH + 'skewed_2_data.npz', sk2_train_x, sk2_train_y, sk2_test_x, sk2_test_y)


def load_data(data_name, args):
    """
    Loads the training and test sets for the given data set.
    """
    DATA_PATH = 'data/' + args.train_dataset + '/'
    target_size = args.target_data_size
    gamma = args.target_test_train_ratio
    with np.load(DATA_PATH + data_name) as f:
        train_x, train_y, test_x, test_y = [f['arr_%d' % i] for i in range(len(f.files))]

    train_x = np.array(train_x, dtype=np.float32)
    test_x = np.array(test_x, dtype=np.float32)

    train_y = np.array(train_y, dtype=np.int32)
    test_y = np.array(test_y, dtype=np.int32)

    return train_x, train_y, test_x[:int(gamma*target_size)], test_y[:int(gamma*target_size)]


def process_features(features, dataset, attribute_dict, max_attr_vals, target_attr, col_flags, skip_sensitive=False, skip_corr=False):
    """
    Returns the feature matrix after expanding the nominal features, 
    and removing the features that are not needed for model training.
    """
    features = pd.DataFrame(features)
    # for removing sensitive feature
    if skip_sensitive:
        features.drop(columns=target_attr, inplace=True)
    # skipping the ST (state) and PUMA attribute for census data set
    if dataset == 'census':
        if col_flags['CENSUS_ST'] != None:
            features.drop(columns=col_flags['CENSUS_ST'], inplace=True)
        if col_flags['CENSUS_PUMA'] != None:
            features.drop(columns=col_flags['CENSUS_PUMA'], inplace=True)
        if col_flags['CENSUS_PINCP'] != None:
            features.drop(columns=col_flags['CENSUS_PINCP'], inplace=True)
    # skipping one of the conflicting attributes for texas data set
    if dataset == 'texas_100_v2':
        if col_flags['TEXAS_THCIC_ID'] != None:
            features.drop(columns=col_flags['TEXAS_THCIC_ID'], inplace=True)
        # skip_corr flag is used to decide whether to skip the correlated attribute in Texas-100X
        if skip_corr == True:
            if col_flags['TEXAS_RACE'] == target_attr:
                features.drop(columns=col_flags['TEXAS_ETHN'], inplace=True)
            elif col_flags['TEXAS_ETHN'] == target_attr:
                features.drop(columns=col_flags['TEXAS_RACE'], inplace=True)
    # for expanding categorical features
    if attribute_dict != None:
        for col in attribute_dict:
            # skip in case the sensitive feature was removed above
            if col not in features.columns:
                continue
            # to expand the non-binary categorical features
            if max_attr_vals[col] != 1:
                features[col] *= max_attr_vals[col]
                features[col] = pd.Categorical(features[col], categories=range(int(max_attr_vals[col])+1))
        features = pd.get_dummies(features)
    return np.array(features)


def threat_model(args, tot_len):
    """
    Returns the data indices based on the adversarial knowledge.
    """
    train_size = args.target_data_size
    test_size = int(args.target_data_size * args.target_test_train_ratio)
    c_size = args.candidate_size
    assert(tot_len >= 4 * (train_size + test_size))
    
    random.seed(args.run) # random.seed(0)
    train_c_idx = random.sample(range(train_size), c_size)
    test_c_idx = random.sample(range(train_size, train_size + test_size), c_size)
    h_test_idx = list(range(2 * train_size + test_size, 2 * (train_size + test_size)))
    sk_test_idx = list(range(3 * train_size + 2 * test_size, 3 * (train_size + test_size)))
    sk2_test_idx = list(range(4 * train_size + 3 * test_size, 4 * (train_size + test_size)))
    
    adv_known_idxs = {}
    # low: adversary only knows data from skewed distribution
    adv_known_idxs['low'] = list(range(2 * (train_size + test_size), 3 * train_size + 2 * test_size))
    # low2: adversary only knows data from skewed_2 distribution
    adv_known_idxs['low2'] = list(range(3 * (train_size + test_size), 4 * train_size + 3 * test_size))
    # med: adversary knows data from training distribution
    adv_known_idxs['med'] = list(range(train_size + test_size, 2 * train_size + test_size))
    #high: adversary knows all train records except candidate set
    adv_known_idxs['high'] = list(set(range(train_size)) - set(train_c_idx))
    
    if tot_len == 5 * (train_size + test_size):
        ood_c_idx = random.sample(range(5 * train_size + 4 * test_size, 5 * (train_size + test_size)), c_size)
        return train_c_idx, test_c_idx, ood_c_idx, h_test_idx, sk_test_idx, sk2_test_idx, adv_known_idxs
    return train_c_idx, test_c_idx, h_test_idx, sk_test_idx, sk2_test_idx, adv_known_idxs


def get_num_classes(dataset_name):
    if dataset_name in ['texas_100_v2', 'purchase_100', 'cifar_100']:
        return 100
    return 2


def get_sensitive_features(dataset_name, data, size=3):
    """
    Returns the sensitive features for a given data set, along 
    with the attribute dictionary if available.
    """
    col_flags = {x: None for x in ['CENSUS_ST', 'CENSUS_PUMA', 'CENSUS_PINCP', 'TEXAS_RACE', 'TEXAS_ETHN', 'TEXAS_THCIC_ID', 'TEXAS_TOTAL_CHARGES']}
    if dataset_name == 'adult':
        attribute_dict, max_attr_vals = pickle.load(open('../dataset/adult_feature_desc.p', 'rb'))
        return [7, 6, 1, 3, 4, 5, 11][:size], attribute_dict, max_attr_vals, col_flags
    elif dataset_name == 'compas':
        attribute_idx, attribute_dict, max_attr_vals = pickle.load(open('../dataset/compas_feature_desc.p', 'rb'))
        return [attribute_idx['sex'], attribute_idx['race'], attribute_idx['age_cat']][:size], attribute_dict, max_attr_vals, col_flags
    elif dataset_name == 'census':
        attribute_idx, attribute_dict, max_attr_vals = pickle.load(open('../dataset/census_feature_desc.p', 'rb'))
        col_flags['CENSUS_ST'] = None if 'ST' not in attribute_idx else attribute_idx['ST']
        col_flags['CENSUS_PUMA'] = None if 'PUMA' not in attribute_idx else attribute_idx['PUMA']
        col_flags['CENSUS_PINCP'] = None if 'PINCP' not in attribute_idx else attribute_idx['PINCP']
        return [attribute_idx['SEX'], attribute_idx['RAC1P'], attribute_idx['DEYE'], attribute_idx['DREM']][:size], attribute_dict, max_attr_vals, col_flags
    elif dataset_name == 'texas_100_v2':
        attribute_idx, attribute_dict, max_attr_vals = pickle.load(open('../dataset/texas_100_v2_feature_desc.p', 'rb'))
        col_flags['TEXAS_TOTAL_CHARGES'] = attribute_idx['TOTAL_CHARGES']
        col_flags['TEXAS_RACE'] = attribute_idx['RACE']
        col_flags['TEXAS_ETHN'] = attribute_idx['ETHNICITY']
        col_flags['TEXAS_THCIC_ID'] = None if 'THCIC_ID' not in attribute_idx else attribute_idx['THCIC_ID']
        return [attribute_idx['SEX_CODE'], attribute_idx['RACE'], attribute_idx['ETHNICITY'], attribute_idx['TYPE_OF_ADMISSION']][:size], attribute_dict, max_attr_vals, col_flags
    return get_random_features(data, list(range(data.shape[1])), size), None, np.ones(data.shape[1], dtype=int), col_flags


def get_random_features(data, pool, size):
    """
    Returns a random set of features from a given data set. 
    These are used for attribute inference, in cases where 
    the sensitive attributes are not known.
    """
    random.seed(SEED)
    features = set()
    while(len(features) < size):
        feat = random.choice(pool)
        c = Counter(data[:, feat])
        # for binary features, select the ones with 1 being minority
        if sorted(list(c.keys())) == [0, 1]:
            if c[1] / len(data) > 0.1 and c[1] / len(data) < 0.5:
                features.add(feat)
        # select feature that has more than one value
        elif len(c.keys()) > 1:
            features.add(feat)
    return list(features)


def subsample(indices, labels, sample_size):
    """
    Returns a subsample of indices such that the sample contains 
    at least one record for each label.
    """
    if sample_size >= len(indices):
        return indices
    idx = []
    labs = np.unique(labels)
    for l in labs:
        idx.append(indices[np.where(np.array(labels) == l)[0][0]])
    random.seed(SEED)
    idx.extend(random.sample(list(set(indices) - set(idx)), sample_size - len(labs)))
    return idx
