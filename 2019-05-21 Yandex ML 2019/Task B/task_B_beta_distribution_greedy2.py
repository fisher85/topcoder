import sys
import numpy as np

# https://habr.com/ru/post/425619/
# Задача о многоруком бандите — сравниваем эпсилон-жадную стратегию и Томпсоновское сэмплирование

# https://web.stanford.edu/~bvr/pubs/TS_Tutorial.pdf

local_test = 1

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
        m = 5
        greedy_n_learning = int(n * 0.1)
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
    
    # А если просто по максимуму наблюдения играть?
    # Не, нужен Томпсон
    nums = np.zeros(m)
    sums = np.zeros(m)
    probs = np.zeros(m)
    eps = np.random.uniform(0, 1, size=n)
    e = 0.1

    # Эпсилон-жадный алгоритм
    for i in range(n):
        if i < greedy_n_learning:
            index = np.random.randint(m)
        else:
            index = np.argmax(probs)

        # epsilon-жадность
        if eps[i] > 1 - e:
            index = np.random.randint(m)            

        print(index+1)
        sys.stdout.flush()

        if local_test == 1:
            reward = np.random.binomial(1, p=payouts[index])
            print(" ", reward)
        else:
            reward = int(input())

        sums[index] += reward
        nums[index] += 1
        probs[index] = sums[index] / nums[index]

    # # j проходов
    # counter2 = 0
    # for j in range(2*m):
    #     for i in range(m):
    #         counter2 += 1
    #         print(i+1)
    #         sys.stdout.flush()
    #         nums[i] += 1
            
    #         if local_test == 1:
    #             reward = np.random.binomial(1, p=payouts[i])
    #             print(" ", reward)
    #         else:
    #             reward = int(input())
    #         sums[i] += reward

    # Набрали статистику, а дальше при каждом запросе сортируем массив наблюдаемых вероятностей выигрыша
    # И играем с теми автоматами, у кого вероятность нашего выигрыша выше

    # probs = sums / nums

    # for i in range(n - counter2):
    #     index = np.argmax(probs)
    #     nums[index] += 1

    #     print(index+1)
    #     sys.stdout.flush()

    #     if local_test == 1:
    #         reward = np.random.binomial(1, p=payouts[index])
    #         print(" ", reward)
    #     else:
    #         reward = int(input())
    #     sums[index] += reward

    #     # Пересчет вероятности выигрыша
    #     probs[index] = sums[index] / nums[index]

    # Учет мат. ожидания?

    print(n, m)
    print("payouts", payouts)
    print("probs", probs)
    print("sums", sums)
    print("sum of sums", np.sum(sums))
    print("sum of nums", np.sum(nums))