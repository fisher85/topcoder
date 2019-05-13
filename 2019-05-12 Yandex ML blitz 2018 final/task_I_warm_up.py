# I. Разминка

import numpy as np

file_name = "2019-05-12 Yandex ML blitz 2018 final/input_I_1_train.tsv"

with open(file_name, "r") as fr:
    data_orig = [[float(num) for num in line.split()] for line in fr]

data_orig = np.asarray(data_orig)

# А если, потренировать классификатор реальный?
# LinearSVC - ничего интересного
from sklearn.svm import LinearSVC
data = data_orig[0:10000,:]
X = data[:,0:-1]
# X = data # - тут сразу правильно срабатывает, показывая самым важным последний столбец :)
y = data[:,-1]

# Логическое ИЛИ пусть будет над первым и третьим столбцом
# Эксперимент удался, опеределяется бесполезный шумовой второй столбец
# X = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]])
# y = np.array([0, 1, 0, 1, 1, 1, 1, 1])
# clf = LinearSVC(random_state=0, tol=1e-5)
# clf.fit(X, y)
# print(clf.coef_)
# print(clf.predict([[1, 1, 0]]))

"""
from sklearn.svm import SVC
clf = SVC(kernel="linear", C=1)
clf.fit(X, y)
print(clf.coef_)
"""

# Дерево решений пробуем
from sklearn.tree import DecisionTreeClassifier
clf = DecisionTreeClassifier(random_state=241)
clf.fit(X, y)
importances = clf.feature_importances_
print(importances)

# Все не то, пробуем feature selection
# https://scikit-learn.org/stable/modules/feature_selection
# 1.13.1. Removing features with low variance
from sklearn.feature_selection import VarianceThreshold
sel = VarianceThreshold(threshold=(.8 * (1 - .8)))
# X = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0], [0, 1, 1], [0, 1, 0], [0, 1, 1]])
new_X = np.array(sel.fit_transform(X))
print(np.sort(sel.variances_))
# Мимо, распределение 0 и 1 в столбце близко к равновероятному
# Threshold = 0.249... = 0.5 * (1 - 0.5)

# Пробуем дальше - sklearn.feature_selection.RFECV
# https://scikit-learn.org/stable/modules/generated/sklearn.feature_selection.RFECV.html#sklearn.feature_selection.RFECV

"""
from sklearn.feature_selection import RFECV
# from sklearn.svm import SVC
# estimator = SVC(kernel="linear", C=1)
estimator = DecisionTreeClassifier(random_state=241)
selector = RFECV(estimator, step=1, cv=5)
new_X = selector.fit_transform(X, y)
print(selector.support_)
print(selector.ranking_)
# Отлично, 6 и 96 столбец - наше всё
"""

# Смотрим на столбцы и подбираем булеву функцию - XOR

# Подготовка решения answer.tsv
input_name = "2019-05-12 Yandex ML blitz 2018 final/input_I_1_test.tsv"
output_name = "2019-05-12 Yandex ML blitz 2018 final/answer.tsv"

with open(input_name, "r") as fr:
    test_data = [[int(num) for num in line.split()] for line in fr]

test_data = np.array(test_data)
# test_data = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])

data_6 = test_data[:,5]
data_96 = test_data[:,95]

# data_6 = test_data[:,0]
# data_96 = test_data[:,1]

result = np.logical_xor(data_6, data_96)
np.savetxt(output_name, result, fmt='%d')