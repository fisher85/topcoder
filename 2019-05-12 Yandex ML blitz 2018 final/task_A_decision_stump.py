# A. Решающий пень

# import numpy as np

# Welford's online algorithm
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm

class Mean():
    def __init__(self):
        self.count = 0.
        self.value = 0.
            
    def Add(self, value):
        self.count += 1.
        self.value += (value - self.value) / self.count

    def Remove(self, value):
        self.count -= 1.
        self.value -= (value - self.value) / self.count

class SumSquare():
    def __init__(self):
        self.mean = Mean()
        self.loss = 0.

    def Add(self, value):
        cur_diff = value - self.mean.value
        self.mean.Add(value)
        self.loss += cur_diff * (value - self.mean.value)

    def Remove(self, value):
        cur_diff = value - self.mean.value
        self.mean.Remove(value)
        self.loss -= cur_diff * (value - self.mean.value)

class Points():
    def __init__(self):
        self.Items = []
        self.Loss = SumSquare()
        self.N = 0

    def Read(self, file_name):
        with open(file_name, "r") as fr:
            for line in fr.readlines()[1:]:
                x, y = map(float, line.strip().split())
                self.Items.append([x, y])
                self.Loss.Add(y)
            self.Items.sort()

    def Solve(self):
        left = SumSquare()
        right = self.Loss

        bestA = 0
        bestB = right.mean.value
        bestC = self.Items[0][0]

        bestQ = right.loss

        for i in range(len(self.Items) - 1):
            item = self.Items[i]
            nextItem = self.Items[i + 1]

            left.Add(item[1])
            right.Remove(item[1])

            if item[0] == nextItem[0]:
                continue

            a = left.mean.value
            b = right.mean.value
            c = (item[0] + nextItem[0]) / 2

            q = left.loss + right.loss

            if q < bestQ:
                bestA = a
                bestB = b
                bestC = c
                bestQ = q

        return bestA, bestB, bestC

file_name = "2019-05-12 Yandex ML blitz 2018 final/input_A_2.txt"
file_name = "stump.in"
points = Points()
points.Read(file_name)
# print("Sorted by x", points.Items)
a, b, c = points.Solve()

output_name = "2019-05-12 Yandex ML blitz 2018 final/output_A_1.txt"
output_name = "stump.out"
with open(output_name, "w") as fw:
    fw.write(str.format("{0:.6f}", a) + " ")
    fw.write(str.format("{0:.6f}", b) + " ")
    fw.write(str.format("{0:.6f}", c))

"""
# Посмотрим
# https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(np.array(points.Items)[:,0], np.array(points.Items)[:,1], "o")
plt.show()
"""   