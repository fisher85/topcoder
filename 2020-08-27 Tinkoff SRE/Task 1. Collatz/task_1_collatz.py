# Найти интервальные суммы чисел
# Гипо́теза Коллатца (3n+1 дилемма, сираку́зская пробле́ма)

import math

l, r = map(int, input().strip().split())
cache = {}
sum = 0
    
for collatz in range(l, r + 1):
    i = collatz
    # print("Очередное число: ", collatz)
    count_steps = 0        

    while i != 1:
        # print("  ", i, count_steps)
        # Если есть в кэше, сразу даем ответ
        if cache.get(i):
            count_steps = cache.get(i) + count_steps
            # print("  Кэш: ", i, int(count_steps))
            break
                
        # Если степень двойки, даем логарифм
        log_of_2 = math.log2(i)
        if log_of_2.is_integer():
            count_steps = log_of_2 + count_steps
            # print("  Логарифм: ", i, int(count_steps))
            break

        # Если четное число
        if i % 2 == 0:
            i = int(i / 2)
            count_steps += 1  
        else: # Нечетное, делаем сразу два шага
            i = int((i * 3 + 1) / 2)
            count_steps += 2

    # print("Полное решение: ", collatz, int(count_steps))
    cache[collatz] = int(count_steps)
    sum += int(count_steps)

print(sum)