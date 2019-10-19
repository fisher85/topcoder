import numpy as np

input_file = "2019-10-19 Yandex ML Contest/input_0.tsv"

with open(input_file, "r") as fr:
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

# Подготовка решения answer.tsv
output_file = "2019-10-19 Yandex ML Contest/answer.tsv"

with open(input_file, "r") as fr:
    test_data = [[float(num) for num in line.split()] for line in fr]

# test_data = np.array([[1, 1], [0, 5]])
result = np.sum(test_data, axis=1)
np.savetxt(output_file, result, fmt='%0.8f')