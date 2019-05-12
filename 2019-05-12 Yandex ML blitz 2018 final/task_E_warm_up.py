# E. Разминка

import numpy as np

file_name = "2019-05-12 Yandex ML blitz 2018 final/input_E_1.txt"
file_name = "2019-05-12 Yandex ML blitz 2018 final/input_E_2_train.tsv"

with open(file_name, "r") as fr:
    data = [[float(num) for num in line.split()] for line in fr]

data = np.asarray(data)
data = data[0:100,:]
a = data[:,0:-1]
b = data[:,-1]

x = np.linalg.solve(a, b)
print(x)

print(a[0,:])
print(format(np.sum(a[0,:]), ".8f"))
print(b[0])

# Хитрецы, все коэффициенты линейных уравнений равны 1

# Подготовка решения answer.tsv

input_name = "2019-05-12 Yandex ML blitz 2018 final/input_E_2_test.tsv"
output_name = "2019-05-12 Yandex ML blitz 2018 final/answer.tsv"

with open(input_name, "r") as fr:
    test_data = [[float(num) for num in line.split()] for line in fr]

# test_data = np.array([[1, 1], [0, 5]])
result = np.sum(test_data, axis=1)
np.savetxt(output_name, result, fmt='%0.8f')