# A. Решающий пень

# Назовём функцию, зависящую от трёх параметров — a, b, c — решающим пнём.

# Сомнительно, что есть аналитический способ решения такой задачи. Решаем перебором.
# Для решения задачи достаточно отсортировать все точки по Х и пытаться построить границу C пня
# между всеми парами точек. Тогда среднее значение Mean для левой части = A, Mean для правой = B.

# Для быстрого пересчета Loss Function = SumSquaredError нужно было найти онлайн-метод расчета ковариации,
# чтобы при перемещении границы не проходить полностью все точки.
# Ковариация отличается от SumSquaredError делением на N. Подойдет первый алгоритм на Википедии - Уэлфорда.

# Welford's online algorithm
# https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Online_algorithm

class MeanCalculator():
    def __init__(self):
        self.count = 0.
        self.mean = 0.

    def Add(self, value):
        self.count += 1.
        self.mean += (value - self.mean) / self.count

    def Remove(self, value):
        self.count -= 1.
        self.mean -= (value - self.mean) / self.count

class SumSquaredErrorCalculator():
    def __init__(self):
        self.meanCalculator = MeanCalculator()
        self.loss = 0.

    def Add(self, value):
        delta = value - self.meanCalculator.mean
        self.meanCalculator.Add(value)
        self.loss += delta * (value - self.meanCalculator.mean)

    def Remove(self, value):
        delta = value - self.meanCalculator.mean
        self.meanCalculator.Remove(value)
        self.loss -= delta * (value - self.meanCalculator.mean)

class Dots():
    def __init__(self):
        self.items = []
        self.overallLoss = SumSquaredErrorCalculator()

    def read(self, file_name):
        with open(file_name, "r") as fr:
            n = int(fr.readline())
            for i in range(0, n):
                x, y = map(int, fr.readline().strip().split())
                self.items.append([x, y])
                self.overallLoss.Add(y)
                print(self.items)
                print("Mean:", self.overallLoss.meanCalculator.mean)
            self.items.sort()
            print("Sorted:", self.items)

    def Solve(self):
        left = SumSquaredErrorCalculator()
        right = self.overallLoss

        bestA = left.meanCalculator.mean
        bestB = right.meanCalculator.mean
        bestC = self.items[0][0]
        bestQ = right.loss

        for i in range(len(self.items) - 1):
            item = self.items[i]
            nextItem = self.items[i + 1]
            print("Split", i, ":", item)

            left.Add(item[1])
            right.Remove(item[1])

            if (item[0] == nextItem[0]):
                continue

            a = left.meanCalculator.mean
            b = right.meanCalculator.mean
            c = (item[0] + nextItem[0]) / 2            

            Q = left.loss + right.loss
            print(Q)

            if Q < bestQ:
                bestA = a
                bestB = b
                bestC = c
                bestQ = Q

        return bestA, bestB, bestC

file_name = "c:\\topcoder\\topcoder\\2020-09-30 Yandex ML Contest\\Warm-up\\input_A_1.txt"
file_name = "D:\\Container\\Python Projects\\2020-09-30 Yandex ML Contest\\Warm-up\\input_A_1.txt"
# file_name = "stump.in"

dots = Dots()
dots.read(file_name)
a, b, c = dots.Solve()

output_name = "c:\\topcoder\\topcoder\\2020-09-30 Yandex ML Contest\\Warm-up\\output_A_1.txt"
output_name = "D:\\Container\\Python Projects\\2020-09-30 Yandex ML Contest\\Warm-up\\output_A_1.txt"
# output_name = "stump.out"

with open(output_name, "w") as fw:
    fw.write(str.format("{0:.6f}", a) + " ")
    fw.write(str.format("{0:.6f}", b) + " ")
    fw.write(str.format("{0:.6f}", c))

# Посмотрим на графике
# https://matplotlib.org/tutorials/introductory/pyplot.html#sphx-glr-tutorials-introductory-pyplot-py

# За прокси: pip install matplotlib --proxy http://stepin_aa:saa@10.0.0.1:3128

print(str.format("{0:.6f}", a) + " " + str.format("{0:.6f}", b) + " " + str.format("{0:.6f}", c))

import numpy as np
import matplotlib.pyplot as plt
fig = plt.figure()
plt.plot(np.array(dots.items)[:,0], np.array(dots.items)[:,1], "bo", markersize=10)
for item in dots.items:
    if item[0] <= c:
        plt.plot(item[0], a, "g*", markersize=5)
    else:
        plt.plot(item[0], b, "r*", markersize=5)
    plt.plot(c, a, "g*", markersize=5)
    plt.plot(c, b, "g*", markersize=5)
plt.show()