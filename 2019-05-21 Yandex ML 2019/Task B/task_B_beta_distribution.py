import sys

while True:
    line1 = input()
    n, m = map(int, line1.strip().split())
    if n == m == 0:
        break
    
    line2 = input()
    alpha, beta = map(float, line2.strip().split())

    # Что нам дает знание распределения? Моменты?
    # mean = alpha / (alpha + beta)

    # А если просто по максимуму наблюдения играть?
    nums = []
    sums = []
    probs = []

    # Два прохода
    for i in range(m):
        print(i+1)
        sys.stdout.flush()
        # nums[i] = 1
        
        line = input()
        sums.append(int(line))

    for i in range(m):
        print(i+1)
        sys.stdout.flush()
        nums.append(2)
        
        line = input()
        sums[i] += int(line)

    # Набрали статистику, а дальше при каждом запросе сортируем массив наблюдаемых вероятностей выигрыша
    # И играем с теми автоматами, у кого вероятность нашего выигрыша выше

    for i in range(m):
        probs.append(sums[i] / nums[i]) # Знаменатель всегда > 0

    for i in range(n-m-m):
        index = probs.index(max(probs))
        nums[index] += 1

        print(index+1)
        sys.stdout.flush()

        line = input()
        sums[index] += int(line)

        # Пересчет вероятности выигрыша
        probs[index] = sums[index] / nums[index]