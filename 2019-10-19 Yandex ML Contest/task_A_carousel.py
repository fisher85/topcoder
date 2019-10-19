import math

# 1. Длина строки 1000, символов 26, можно прямым перебором, начиная с длины = 1
# 2. На очередной итерации проверки делим всю строку на подстроки длины i. 
# Последняя подстрока может быть меньшей длины, при сравнении это нужно учитывать.
# 3. Начинаем сравнивать ПЕРВУЮ подстроку со всеми остальными в строке.
# 4. Если ПЕРВАЯ подстрока содержит неопределенности, которые однозначно разрешаются при очередном сравнении,
# например, #bc и abc, неопределенность в ПЕРВОЙ подстроке разрешаем и сравниваем далее уже в таком виде.
# 5. Если неопределенность в правой подстроке, оставляем ее.
# 6. Минимум = 1, максимум = не 26, а n

yandex = False
input_file = "2019-10-19 Yandex ML Contest/input.txt"
input_file = "input.txt"

with open(input_file, "r") as fr:
    input_lines = fr.readlines()
    loop_length = math.floor(len(input_lines) / 2)
    for x in range(0, loop_length):
        # Для тестирования во входном файле записываем пачку тестов
        n = int(input_lines[0 + x*2])
        s = input_lines[1 + x*2].strip()
        success = False

        # if (yandex == False): print("-------------------------------------------- Новый тест:", n, s)

        # Перебор минимальной длины
        for sub_len in range(1, n+1):
            # if (yandex == False): print("Итерация длины подстроки:", sub_len, s, len(s))
            
            # Делим строку на подстроки
            subs = []
            subs_count = math.ceil(len(s) / sub_len)
            # if (yandex == False): print("Число подстрок:", subs_count)
            for j in range(0, subs_count):
                sub_start_index = 0 + sub_len*j
                sub_finish_index = sub_start_index + sub_len
                # 2
                if (sub_finish_index > len(s)): sub_finish_index = len(s)
                subs.append(s[sub_start_index:sub_finish_index])
            # if (yandex == False): print(*subs)

            # Начинаем посимвольно сравнивать ПЕРВУЮ подстроку со всеми остальными
            equal = True
            left = list(subs[0])
            for sub_index in range(1, subs_count):
                right = list(subs[sub_index])
                # if (yandex == False): print("Сравнение подстрок:", *left, *right)
                for position in range(0, sub_len):
                    # 2
                    if (position >= len(right)): break
                    if (left[position] == '#'):
                        if (right[position] == '#'): continue
                        # 4
                        left[position] = right[position]
                        continue
                    if (right[position] == '#'): continue;
                    if (left[position] != right[position]): 
                        equal = False
                        break
                # Если очередная пара не совпадает, сразу переходим на следующую итерацию длины
                if (equal == False): break
                    
            if (equal == True):
                # Флаг сохранился после всех попарных сравнений, значит это ответ
                # if (yandex == False): print("Ответ:", sub_len)
                print(sub_len)
                break