# E. Разминка

import numpy as np

file_name = "2019-05-12 Yandex ML blitz 2018 final/input_E_1.txt"
file_name = "2019-05-12 Yandex ML blitz 2018 final/input_E_2_train.tsv"

with open(file_name, "r") as fr:
    data_orig = [[float(num) for num in line.split()] for line in fr]

data_orig = np.asarray(data_orig)
data = data_orig[0:100,:]
a = data[:,0:-1]
b = data[:,-1]

x = np.linalg.solve(a, b)
print(x)

print(a[0,:])
print(format(np.sum(a[0,:]), ".8f"))
print(b[0])

# Хитрецы, все коэффициенты линейных уравнений равны 1

# А если, потренировать классификатор реальный?
# Получится медленнее, нужно будет все 10000 использовать при обучении, погрешность 1e-10
from sklearn.svm import LinearSVR
data = data_orig[0:10000,:]
X = data[:,0:-1]
y = data[:,-1]
clf = LinearSVR(random_state=0, tol=1e-10)
clf.fit(X, y)
print(clf.coef_)

# Подготовка решения answer.tsv

input_name = "2019-05-12 Yandex ML blitz 2018 final/input_E_2_test.tsv"
output_name = "2019-05-12 Yandex ML blitz 2018 final/answer.tsv"

with open(input_name, "r") as fr:
    test_data = [[float(num) for num in line.split()] for line in fr]

# test_data = np.array([[1, 1], [0, 5]])
result = np.sum(test_data, axis=1)
np.savetxt(output_name, result, fmt='%0.8f')