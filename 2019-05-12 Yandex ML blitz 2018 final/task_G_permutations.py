# I. Перестановки

import random
import numpy as np

def RandomPermutation():  
    perm = list(range(8))  
    random.shuffle(perm)  
    return perm  
 
def StupidPermutation():  
    partialSums = [0,1,8,35,111,285,  
        628,1230,2191,3606,5546,8039,11056,14506,18242,  
        22078,25814,29264,32281,34774,36714,38129,39090,  
        39692,40035,40209,40285,40312,40319,40320]  
    r = random.randint(0, partialSums[-1])  
    numInv = 0  
    while partialSums[numInv] < r:  
        numInv += 1  
    perm = list(range(8))  
    for step in range(numInv):  
        t1 = random.randint(0, 7)  
        t2 = random.randint(0, 7)  
        perm[t1], perm[t2] = perm[t2], perm[t1]  
    return perm

def Permutation():
    return StupidPermutation()

# Неудачный вариант
def score_std(dataset_scored):
    score = 0
    for i in range(8):
        unique_elements, counts_elements = np.unique(dataset_scored[:,i], return_counts=True)
        score += np.std(counts_elements)
    return score

# Вот эта оценка прошла тест
def score_mean(dataset_scored):
    score = 0
    sum_diagonal = 0
    sum_overall = 0
    for i in range(8):
        unique_elements, counts_elements = np.unique(dataset_scored[:,i], return_counts=True)
        sum_diagonal += counts_elements[i]
        sum_overall += np.sum(counts_elements)
    mean_diagonal = sum_diagonal / 8
    mean_overall = sum_overall / 64
    return mean_diagonal - mean_overall

# Для начала оценим равновероятность появлений чисел в столбцах
# Быстро и грубо - суммы в столбцах посмотреть

dataset = []
for i in range(1000):
    dataset.append(Permutation())
dataset = np.asarray(dataset)
print(np.sum(dataset, axis=0))
# Плохо, суммы пляшут

# Посмотрим на распределение частот
for i in range(8):
    unique_elements, counts_elements = np.unique(dataset[:,i], return_counts=True)
    print("Frequency of unique values of the ", i, " column:")
    print(np.asarray((unique_elements, counts_elements)))
    print(counts_elements)
    print(np.std(counts_elements))
# Для StupidPermutation есть явные всплески, особенно диагональные на 10000

# Коротко оформляем в виде score_std
print(score_std(dataset))

file_name = "2019-05-12 Yandex ML blitz 2018 final/permutations.in"

with open(file_name, "r") as fr:
    input_lines = fr.readlines()
    n = int(input_lines[0])
    data_orig = [[int(num) for num in line.split()] for line in input_lines[1:]]

data_orig = np.asarray(data_orig)
scores = []

for set_index in range(n):
    print('SET', set_index+1)
    shift = 1000 * set_index
    dataset = data_orig[shift:shift+999,:]
    scores.append((set_index, score_mean(dataset)))

print(scores)
scores = np.asarray(scores)

# Формируем ответ
answer = scores[np.argsort(scores[:,1])]
answer_indexes = answer[:,0]
output_name = "2019-05-12 Yandex ML blitz 2018 final/task_G.out"
np.savetxt(output_name, answer_indexes, fmt='%d')