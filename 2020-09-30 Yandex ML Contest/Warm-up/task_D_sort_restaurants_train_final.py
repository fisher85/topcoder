file_in = "c:\\topcoder\\topcoder\\2020-09-30 Yandex ML Contest\\Warm-up\\restaurants1.in"

import sys
sys.stdin = open(file_in, "r", encoding="utf-8")

from decimal import *

def score(r, d, coefs = [Decimal(1), Decimal(1), Decimal(0)]):
    result = coefs[0] * r + coefs[1] * d + coefs[2]
    return result

n = int(input())
for i in range(0, n):
    r, d = map(Decimal, input().split())
    if not d.is_normal(): d = Decimal(1)
    result = score(r, d)
    print(str.format("{0:.12f}", result))