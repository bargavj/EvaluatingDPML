#used to normalize and set up the raw data into pickle files

import pickle
import numpy as np
from sklearn.preprocessing import normalize

def normalizeDataset(x):
    mods = np.linalg.norm(x, axis=1)
    return x/mods[:, np.newaxis]

with open('train', 'rb') as fo:
    dict = pickle.load(fo, encoding='bytes')

pickle.dump(normalizeDataset(dict[b'data']), open('', 'wb'))
pickle.dump(dict[b'fine_labels'], open('', 'wb'))
