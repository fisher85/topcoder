# You should use fast Python 2.7
n = input()
previous = -10000000000

for i in range(0, n):
    current = input()
    if (current > previous):
        previous = current
        print(current)

# Python 3 is slow, test 193 => 1.085 sec
# n = int(input())
# previous = int(-10000000000)

# for i in range(0, n):
#     current = int(input())
#     if (current > previous):
#         previous = current
#         print(current)