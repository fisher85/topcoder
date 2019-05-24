import sys
import numpy as np

# https://habr.com/ru/post/425619/
# Задача о многоруком бандите — сравниваем эпсилон-жадную стратегию и Томпсоновское сэмплирование

# https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf

local_test = 0
counter = 0
max_counter = 1

while True:
    counter += 1

    if local_test == 1:
        if counter > max_counter: 
            break
        n = np.random.randint(200, 501)
        m = np.random.randint(5,101)
        n = 500
        m = 50
        alpha = 2
        beta = 2
        payouts = np.random.beta(alpha, beta, size=m)
    else:
        line1 = input()
        n, m = map(int, line1.strip().split())
        if n == 0 and m == 0:
            break
        line2 = input()
        alpha, beta = map(float, line2.strip().split())

    # Сэмплирование Томпсона
    a = np.ones(m)
    b = np.ones(m)
    #for i in range(n):
     #   a[i] = int(alpha)
      #  b[i] = int(beta)

    nums = np.zeros(m)

    for i in range(n):
        probs = np.random.beta(a, b)
        index = np.argmax(probs)

        print(index+1)
        sys.stdout.flush()

        if local_test == 1:
            reward = np.random.binomial(1, p=payouts[index])
            print(" ", reward)
        else:
            reward = int(input())

        a[index] += reward
        b[index] += 1 - reward 
        nums[index] += 1

    # print(n, m)
    # print("payouts", payouts)
    # print("probs", probs)
    # print("nums", nums)
    # print("total reward", np.sum(a) - m)