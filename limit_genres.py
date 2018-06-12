#!/bin/usr/env python3

import pandas as pd
import numpy as np

file = pd.read_csv('full_dataset.csv')
data = file.values
row, col = data.shape
names = np.array(
        ['bandwidth', 'pervasive_freq', 'signal_strength', 'signal_envelope'])
Xs, ys = [], []
data_3gen = pd.DataFrame()
for i in range(row):
    if data[i, 3] <= 3:
        X = data[i, 4:]
        Xs.append(X)
        y = data[i, 3]
        ys.append(y)
Xs = pd.DataFrame(np.array(Xs), columns=names)
ys = np.transpose(np.array(ys))
ys = pd.DataFrame({'genre': ys})
data_3gen = ys.join(Xs, how='right')
data_3gen.to_csv('three_genre_dataset.csv')
