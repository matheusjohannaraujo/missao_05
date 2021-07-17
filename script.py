
"""

https://scikit-learn.org/stable/modules/svm.html
https://scikit-learn.org/stable/modules/multiclass.html
https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RepeatedKFold.html
https://archive.ics.uci.edu/ml/datasets/glass+identification

"""

import matplotlib
import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
import sklearn

dataset_df = pd.read_csv('glass.data')
dataset_df.columns = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k']

classes_sr = dataset_df['k']

#print(classes_sr)

k = []
for i in range(1, 8):
    k.append(dataset_df.loc[dataset_df['k'] == i])
print(k)

#scaler = preprocessing.StandardScaler().fit(classes_sr)

#print(scaler)

#print(dataset.loc[10])

"""dataset2 = np.loadtxt('glass.data', delimiter=',')

#print(dataset.head())

#print(dataset)

#print(dataset2[0])

x = dataset2[:,0:7]
y = dataset2[:,8]

print(y)

#print(dataset.loc[0])
"""