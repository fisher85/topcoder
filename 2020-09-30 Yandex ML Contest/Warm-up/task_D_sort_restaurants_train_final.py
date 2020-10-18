# Решение на 2.29527195132 балла из 4 возможных

file_in = "D:\\Container\\Python Projects\\2020-09-30 Yandex ML Contest\\Warm-up\\restaurants1.in"

import sys
sys.stdin = open(file_in, "r", encoding="utf-8")

def score(r, d, coefs = [0.12868700000001043, -3.60570000000003, 0.30492999999999837]):
    result = (coefs[0] * r) + (coefs[1] * d) + (coefs[2] * r * d)
    return result

n = int(input())
for i in range(0, n):
    r, d = map(float, input().strip().split("\t"))
    print(str.format("{0:.12f}", score(r, d)))