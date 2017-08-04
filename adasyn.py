import pickle
import logging
import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from imblearn.over_sampling import ADASYN

print('Loading creditcard.csv')
data = pd.read_csv("data/creditcard.csv")
#Create dataframes of only Fraud and Normal transactions. Also Shuffle them.
fraud = shuffle(data[data.Class == 1])
normal = shuffle(data[data.Class == 0])
# Produce a training set of 80% of fraudulent and 80% normal transactions
X_train = fraud.sample(frac=0.8)
X_train = pd.concat([X_train, normal.sample(frac = 0.8)], axis = 0)

ada = ADASYN(n_jobs=3)
ada.logger.error('test')
ada.logger.setLevel(logging.INFO)

print('starting...')

data_resampled, data_labels_resampled = ada.fit_sample(
                                            np.array(X_train.ix[:, X_train.columns != 'Class']), 
                                            np.array(X_train.Class))

print('pickling...')
with open('pickle/train_data_resampled.pkl', 'wb+') as f:
    pickle.dump(data_resampled, f)

with open('pickle/train_data_labels_resampled.pkl', 'wb+') as f:
    pickle.dump(data_labels_resampled, f)