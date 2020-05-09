import pickle
import operator
import numpy as np
from sklearn.preprocessing import normalize
from sklearn.cluster import KMeans

IT_NUM = 100

def normalizeDataset(X):
    mods = np.linalg.norm(X, axis=1)
    return X / mods[:, np.newaxis]

def populate1():    
    cnt = 0
    items = dict()
    customer = dict()
    fp = open('transactions.csv')
    for line in fp:
        cnt += 1
        if cnt == 1:
            continue
        cust = line.split(',')[0]
        it = line.split(',')[3]
        if it not in items:
            items[it] = set(cust)
        else:
            items[it].add(cust)
    for key, val in items.items():
        items[key] = len(val)
    sorted_items = sorted(items.items(), key=operator.itemgetter(1))
    freq_items = [tup[0] for tup in sorted_items[-IT_NUM:]]
    cnt = 0
    freq_items_dict = dict()
    for it in freq_items:
        freq_items_dict[it] = cnt
        cnt += 1
    print(freq_items, len(freq_items))

    cnt = 0
    fp = open('transactions.csv')
    for line in fp:
        cnt += 1
        if cnt == 1:
            continue
        cust = line.split(',')[0]
        it = line.split(',')[3]
        if it in freq_items:
            if cust not in customer:
                customer[cust] = [0]*IT_NUM
            customer[cust][freq_items_dict[it]] = 1

    print(len(customer))
    pickle.dump([customer, freq_items_dict], open('transactions_dump.p', 'wb'))

def populate():    
    fp = open('transactions.csv')
    cnt, cust_cnt, it_cnt = 0, 0, 0
    items = dict()
    customer = dict()
    last_cust = ''
    for line in fp:
        cnt += 1
        if cnt == 1:
            continue
        cust = line.split(',')[0]
        it = line.split(',')[3]
        if it not in items and it_cnt < IT_NUM:
            items[it] = it_cnt
            it_cnt += 1
        if cust not in customer:
            customer[cust] = [0]*IT_NUM
            cust_cnt += 1
            last_cust = cust
        if cust_cnt > 250000:
            break
        if it in items:
            customer[cust][items[it]] = 1
        if cnt % 10000 == 0:
            print(cnt, cust_cnt, it_cnt)
    del customer[last_cust]
    print(len(customer), len(items))

    no_purchase = []
    for key, val in customer.items():
    	if 1 not in val:
    		no_purchase.append(key)
    for cus in no_purchase:
	    del customer[cus]
    print(len(customer), len(items))

    pickle.dump([customer, items], open('transactions_dump.p', 'wb'))

def make_dataset():
    dataset = []
    customer, items = pickle.load(open('transactions_dump.p', 'rb'))
    for key, val in customer.items():
        dataset.append(val)
    dataset = np.array(dataset)
    dataset = normalizeDataset(dataset)
    print(dataset.shape)
    pickle.dump(dataset, open('purchase_100_features.p', 'wb'))
    X = KMeans(n_clusters=100, random_state=0).fit(dataset)
    pickle.dump(X.labels_, open('purchase_100_labels.p', 'wb'))
    print(np.unique(X.labels_))

# Note: transactions.csv file can be downloaded from https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data
#populate1() # 100 'most' frequent items
populate() # first 100 frequent items -- generates the data set used in the experiments
make_dataset()