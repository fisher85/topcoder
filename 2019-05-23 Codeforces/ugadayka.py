# https://codeforces.com/gym/101021/problem/A
import sys

left = 1
right = 1000000

while True:
    value = int((left + right + 1) / 2)
    print(value)
    sys.stdout.flush()

    line = input()
    if line == ">=":
        left = value
    else:
        right = value - 1

    if right == left:
        print("!", right)
        break