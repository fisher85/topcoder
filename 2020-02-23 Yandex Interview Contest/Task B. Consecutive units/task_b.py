n = int(input())
max = 0
current = 0

for i in range(0, n):
    digit = int(input())
    if (digit == 1):
        current = current + 1
        if (current > max):
            max = current
    if (digit == 0):
        current = 0

print(max)