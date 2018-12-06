#!/usr/bin/env python3

"""Uczenie i testowanie prównawczych algorytmów klasyfikacji"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import neighbors
from sklearn import svm
from sklearn import linear_model
from sklearn import neural_network
from sklearn import model_selection
from sklearn import metrics
from scipy import stats
from tqdm import tqdm
import itertools

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Prawdziwe etykiety')
    plt.xlabel('Przewidywane etykiety')
    plt.tight_layout()

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
# k-foldowa walidacja krzyżowa
k_means = []
ks = [10]
for k in ks:
    cv = model_selection.StratifiedKFold(n_splits=k)
    results, k_pars, C_pars, acc_kNN, acc_SVM, predis, trues = [], [], [], [], [], [], []
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
            if True:

                clf.fit(X_train, y_train)
                support_vector = clf.predict_proba(X_test)
                prediction = np.argmax(support_vector, axis=1)
                prediction = clf.classes_[prediction]

                accuracy = metrics.accuracy_score(y_test, prediction)
                accuracies.append(accuracy)

                if clf == clfs[0]:
                    # k_params = clf.get_params()['n_neighbors']
                    # k_pars.append(k_params)
                    trues.extend(y_test)
                    predis.extend(prediction)
                '''                  
                if clf == clfs[1]:
                    C_params = clf.get_params()['C']
                    C_pars.append(C_params)
                '''
                results.append(accuracies)

    # print('lista ilosc sasiadow z kNN', k_pars)
    # print('lista wartosci kosztow z SVM', C_pars)
    results = np.array(results)
    names = np.array(['kNN', 'SVM', 'LR', 'MLP'])
    datas = pd.DataFrame(results, columns=names)
    means = []
    print('\nAverage accuracy of classifiers for ' + str(k) + '-fold cross-validation:')
    for name in names:
        mean = np.mean(datas.loc[:, name])
        means.append(mean)
        print('\t', name, ': %.2f' % (mean * 100), '%')
        # print(name, datas.loc[:, name])
    means = [one * 100 for one in means]
    k_means.append(means)
    max_acc = max(means)
    avg_acc = {names[0]: [means[0], 0], names[1]: [means[1], 1],
               names[2]: [means[2], 2], names[3]: [means[3], 3]}

    predis = np.array(predis)
    conf_mx = metrics.confusion_matrix(trues, predis, labels=['Rock', 'Hiphop', 'Classical', 'Blues', 'Country',
                                                              'Reggae', 'Disco', 'Pop', 'Jazz', 'Metal'])
    plt.figure()
    plot_confusion_matrix(conf_mx, classes=['Rock', 'Hiphop', 'Classical', 'Blues', 'Country',
                                                              'Reggae', 'Disco', 'Pop', 'Jazz', 'Metal'],
                          title='Macierz pomyłek dla kNN')
    # plt.savefig('results/conf_mx_MLP.eps', format='eps')
    # plt.show()

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
'''
fig, ax = plt.subplots()
bar_width = 0.35
index = np.arange(len(names))
means_k_10 = ax.bar(index, k_means[0], bar_width, label='10-krotna walidacja krzyżowa')
means_k_30 = ax.bar(index + bar_width, k_means[1], bar_width, label='30-krotna walidacja krzyżowa')
ax.set_xlabel('Klasyfikatory')
ax.set_ylabel('Dokładności procentowe')
ax.set_title('Uzyskane średnie dokładości')
ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(tuple(names))
ax.legend()
fig.tight_layout()
# plt.savefig('results/kFold_comp.svg')
plt.show()

    cnf_matrix = conf_mx
    FP = cnf_matrix.sum(axis=0) - np.diag(cnf_matrix)
    FN = cnf_matrix.sum(axis=1) - np.diag(cnf_matrix)
    TP = np.diag(cnf_matrix)
    TN = cnf_matrix.sum() - (FP + FN + TP)
    FP = FP.astype(float)
    FN = FN.astype(float)
    TP = TP.astype(float)
    TN = TN.astype(float)
    # Sensitivity, hit rate, recall, or true positive rate
    TPR = TP / (TP + FN)
    # Specificity or true negative rate
    TNR = TN / (TN + FP)
    # Precision or positive predictive value
    PPV = TP / (TP + FP)
    # Negative predictive value
    NPV = TN / (TN + FN)
    # Fall out or false positive rate
    FPR = FP / (FP + TN)
    # False negative rate
    FNR = FN / (TP + FN)
    # False discovery rate
    FDR = FP / (TP + FP)
    # Overall accuracy
    ACC = (TP + TN) / (TP + FP + FN + TN)
    print(TPR)
    print(FPR)
'''
