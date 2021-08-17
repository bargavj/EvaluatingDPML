import numpy as np
import pickle

DATA_PATH = '../dataset/'
data_set = 'purchase_100'

X = pickle.load(open(DATA_PATH+data_set+'_features.p', 'rb'))
y = pickle.load(open(DATA_PATH+data_set+'_labels.p', 'rb'))

print(X, X.shape)
X = np.where(X != 0, 1, 0)
print(X, X.shape)

print(y, y.shape)

pickle.dump(X, open(DATA_PATH+data_set+'_binary_features.p', 'wb'))
pickle.dump(y, open(DATA_PATH+data_set+'_binary_labels.p', 'wb'))
