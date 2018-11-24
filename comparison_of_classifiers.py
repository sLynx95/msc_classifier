#!/usr/bin/env python3

"""Uczenie i testowanie prównawczych algorytmów klasyfikacji"""

import pandas as pd
import numpy as np
from sklearn import neighbors
from sklearn import svm
from sklearn import linear_model
from sklearn import neural_network
from sklearn import model_selection
from sklearn import metrics
from scipy import stats
from tqdm import tqdm

pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

FILE_NAME = 'datasets/dataset_GZN_PCA_standardized.csv'

# Czytamy CSV i dokonujemy podziału na cechy i etykiety
df = pd.read_csv(FILE_NAME)
data = df.values
# Ustawiono częściowy zbiór
X = np.array(data[:, :4])
y = np.array(data[:, 5])
# print(X.shape, y.shape)

# Hiperparametry dla SVM
svm_params = {'probability': [True], 'C': [1.0, 10.0, 100.0, 1000.0]}
# Hiperparametry dla kNN
kNN_params = {'n_neighbors': [1, 3, 5, 7, 10]}
search = [model_selection.GridSearchCV(svm.SVC(), svm_params, cv=None),
            model_selection.GridSearchCV(neighbors.KNeighborsClassifier(), kNN_params, cv=None)]
# 30-foldowa walidacja krzyżowa
k = 30
cv = model_selection.StratifiedKFold(n_splits=k)
results = []
for train, test in tqdm(cv.split(X, y), desc='Execute by folds', total=k):
    X_train, y_train = X[train], y[train]
    X_test, y_test = X[test], y[test]
    for idx in tqdm(range(2), desc='Searching hyper-params'):
        search[idx].fit(X_train, y_train)
        if idx == 0:
            best_C, prop = search[idx].best_params_['C'], search[idx].best_params_['probability']
        elif idx == 1:
            best_k = search[idx].best_params_['n_neighbors']
    # Cztery różne klasyfikatory
    clfs = [
        neighbors.KNeighborsClassifier(n_neighbors=best_k),
        svm.SVC(probability=prop, C=best_C),
        linear_model.LogisticRegression(),
        neural_network.MLPClassifier()
    ]
    accuracies = []
    for clf in tqdm(clfs, desc='Execute by classifiers', total=len(clfs)):
        clf.fit(X_train, y_train)
        support_vector = clf.predict_proba(X_test)
        prediction = np.argmax(support_vector, axis=1)
        prediction = clf.classes_[prediction]
        accuracy = metrics.accuracy_score(y_test, prediction)
        accuracies.append(accuracy)
    results.append(accuracies)
print()
results = np.array(results)

names = np.array(['kNN', 'SVM', 'LR', 'MLP'])
datas = pd.DataFrame(results, columns=names)
means = []
print('Average accuracy of classifiers:')
for name in names:
    mean = np.mean(datas.loc[:, name])
    means.append(mean)
    print('\t', name, ': %.2f' % (mean * 100), '%')
max_acc = max(means)
avg_acc = {names[0]: [means[0], 0], names[1]: [means[1], 1],
           names[2]: [means[2], 2], names[3]: [means[3], 3]}
# Testy T-studenta najwyższa dokładność vs reszta
for key, value in avg_acc.items():
    if max_acc == avg_acc[key][0]:
        print('Statical T-student tests -> p-value:')
        if value[1] == 0:
            test_t1 = stats.ttest_ind(results[:, 0], results[:, 1])
            test_t2 = stats.ttest_ind(results[:, 0], results[:, 2])
            test_t3 = stats.ttest_ind(results[:, 0], results[:, 3])
            print('\t%s vs %s : %.3f' % (key, names[1], test_t1.pvalue))
            print('\t%s vs %s : %.3f' % (key, names[2], test_t2.pvalue))
            print('\t%s vs %s : %.3f' % (key, names[3], test_t3.pvalue))
        elif value[1] == 1:
            test_t1 = stats.ttest_ind(results[:, 1], results[:, 0])
            test_t2 = stats.ttest_ind(results[:, 1], results[:, 2])
            test_t3 = stats.ttest_ind(results[:, 1], results[:, 3])
            print('\t%s vs %s : %.3f' % (key, names[0], test_t1.pvalue))
            print('\t%s vs %s : %.3f' % (key, names[2], test_t2.pvalue))
            print('\t%s vs %s : %.3f' % (key, names[3], test_t3.pvalue))
        elif value[1] == 2:
            test_t1 = stats.ttest_ind(results[:, 2], results[:, 0])
            test_t2 = stats.ttest_ind(results[:, 2], results[:, 1])
            test_t3 = stats.ttest_ind(results[:, 2], results[:, 3])
            print('\t%s vs %s : %.3f' % (key, names[0], test_t1.pvalue))
            print('\t%s vs %s : %.3f' % (key, names[1], test_t2.pvalue))
            print('\t%s vs %s : %.3f' % (key, names[3], test_t3.pvalue))
        elif value[1] == 3:
            test_t1 = stats.ttest_ind(results[:, 3], results[:, 0])
            test_t2 = stats.ttest_ind(results[:, 3], results[:, 1])
            test_t3 = stats.ttest_ind(results[:, 3], results[:, 2])
            print('\t%s vs %s : %.3f' % (key, names[0], test_t1.pvalue))
            print('\t%s vs %s : %.3f' % (key, names[1], test_t2.pvalue))
            print('\t%s vs %s : %.3f' % (key, names[2], test_t3.pvalue))
