# A. Уникальные запросы

# Пользователи задают в Яндекс.Поиске десятки тысяч запросов в секунду. 
# Часть запросов задают сотни раз в час, другая часть запросов повторяется несколько раз в день, 
# третью часть запросов пользователи спрашивают у Яндекса впервые.

# Необходимо оценить количество уникальных запросов, при условии наличия 500 KB оперативной памяти. 
# Гарантируется, что правильный ответ не превосходит 100000 и не меньше, чем 50000.

# Решение засчитывается, если ответ отличается от правильного не более, чем на 5%.

# Первое решение - в лоб, baseline.

file_name = "c:\\topcoder\\topcoder\\2020-09-30 Yandex ML Contest\\Warm-up\\input_C_1.txt"
# file_name = "stump.in"

import sys
sys.stdin = open(file_name, "r", encoding="utf-8")

"""
queries = set()

n = int(input())
for i in range(0, n):
    query = input().strip()
    # print(query)
    queries.add(query)

print(len(queries))
"""

# Ага, решение в лоб не проходит по памяти, начиная со второго теста :)

# Решение в том, что нужно применить любой вероятностный метод решения задачи
# https://en.wikipedia.org/wiki/Count-distinct_problem

# The HyperLogLog algorithm is able to estimate cardinalities of > 10^9 
# with a typical accuracy (standard error) of 2%, using 1.5 kB of memory.
# https://github.com/clarkduvall/hypy - отсюда взята реализация HyperLogLog
# https://habr.com/ru/post/119852/
# https://github.com/svpcom/hyperloglog

# Почему были проблемы с подходом хранения полного хэша запроса? Если нам по условию
# хватает точно 100000 сохраненных хэшей, то можно было ожидать, что по 4 байта на каждый хэш
# хватит памяти. Однако python для типа int занимает чуть ли нет 24 байта, а set() вообще огромный!

from math import log

class HLL(object):
    P32 = 2 ** 32

    def __init__(self, p=14):
        self.p, self.m, self.r = p, 1 << p, [0] * (1 << p)

    def add(self, x):
        x = hash(x)
        i = x & HLL.P32 - 1 >> 32 - self.p
        z = 35 - len(bin(HLL.P32 - 1 & x << self.p | 1 << self.p - 1))
        self.r[i] = max(self.r[i], z)

    def count(self):
        a = ({16: 0.673, 32: 0.697, 64: 0.709}[self.m]
             if self.m <= 64 else 0.7213 / (1 + 1.079 / self.m))
        e = a * self.m * self.m / sum(1.0 / (1 << x) for x in self.r)
        if e <= self.m * 2.5:
            z = len([r for r in self.r if not r])
            return int(self.m * log(float(self.m) / z) if z else e)
        return int(e if e < HLL.P32 / 30 else -HLL.P32 * log(1 - e / HLL.P32))

hll = HLL()
    
n = int(input())
for i in range(0, n):
    query = input().strip()
    hll.add(query)

print(hll.count())