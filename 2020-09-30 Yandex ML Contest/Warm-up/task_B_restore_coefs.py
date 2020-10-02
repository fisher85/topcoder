# A. Уникальные запросы

# Необходимо восстановить коэффициенты функции f(x), зная её значения на некотором наборе точек.
# При этом известно, что f(x) = ((a + ea) sin x + (b + eb) ln x) ^ 2 + (c + ec) x ^ 2
# где ea, eb, ec — случайные величины, которые принимают значения из отрезка [–0.001, 0.001]; 
# a, b, c — неизвестные положительные константы, которые требуется найти (абсолютная ошибка не должна превышать 10-2).

import math
import numpy as np

file_input = "c:\\topcoder\\topcoder\\2020-09-30 Yandex ML Contest\\Warm-up\\input_B.csv"
file_output = "c:\\topcoder\\topcoder\\2020-09-30 Yandex ML Contest\\Warm-up\\output_B.txt"

points = []

with open(file_input, "r") as fr:
    input_lines = fr.readlines()
    for line in input_lines:
        x, y = map(float, line.strip().split(","))
        points.append([x, y])

points = sorted(points, key=lambda k: k[0])

def get_loss(points, a, b, c):
    mse = 0.
    for point in points:
        x = point[0]
        y = point[1]
        fx = (a * math.sin(x) + b * math.log(x)) ** 2 + c * (x ** 2)
        mse += (y - fx) ** 2
    return mse / len(points)

"""
with open(file_output, "w") as fw:
    for point in points:
        fw.write("(" + str.format("{0:.10f}", point[0]) + ";" + str.format("{0:.10f}", point[1]) + ") ")
"""

min_loss = 9999999999.
bestA = 0.
bestB = 0.
bestC = 0.

max = 10
steps = 10

for a in np.linspace(math.pi - 0.005, math.pi + 0.005, steps):
    print(round(a), bestA, bestB, bestC, min_loss)
    for b in np.linspace(math.exp(1) - 0.005, math.exp(1) + 0.005, steps):
        c = 4.0
        # for c in np.linspace(3, 4, steps):
        loss = get_loss(points, a, b, c)
        if loss < min_loss:
            min_loss = loss
            bestA = a
            bestB = b
            bestC = c

print(bestA, bestB, bestC, min_loss)
print(math.pi, math.exp(1), 4.0, get_loss(points, math.pi, math.exp(1), 4.0))