#!/usr/bin/env python3

"""Standaryzacja i zastosowanie PCA na zbiorze danych """

import os
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

os.chdir('datasets/')
DATASETPATH = '/home/seba/dev/msc_classifier/datasets/oryginal_full_dataset_GZN.csv'
NAME_FEATURES = np.array(
    ['bandwidth', 'pervasive_freq', 'signal_strength', 'signal_envelope'])


# Czytamy CSV i dokonujemy podziału na cechy i etykiety
df = pd.read_csv(DATASETPATH)
data = df.values
X = np.array(data[:, 1:])
y = np.array(data[:, 0])
y = pd.DataFrame(y, columns=['genre'])

# Standaryzacja cech z utworu
scaler = StandardScaler()
scaled_X = pd.DataFrame(scaler.fit_transform(X, y), columns=NAME_FEATURES)
scaled_dataset = scaled_X.join(y)
# scaled_dataset.to_csv('dataset_GZN_standardized.csv')
print(scaled_dataset)

# Zastosowanie analizy główych składowych - PCA
pca = PCA()
after_PCA = pd.DataFrame(pca.fit_transform(scaled_X, y), columns=NAME_FEATURES)
dataset_after_PCA = after_PCA.join(y)
# dataset_after_PCA.to_csv('dataset_GZN_PCA_standardized.csv')
print(dataset_after_PCA)
