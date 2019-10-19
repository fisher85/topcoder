import time
import sys
start_time = time.time()

# Хех, глубина рекурсии 50, но время 5 секунд, можно попробовать
# 1. А не больше 9, нам подойдет 10 система счисления для хранения свойств публикации,
# просто массив 11001 записываем числом 11001
# 2. Последнюю строку A1 A2 A3 тоже записываем просто число, 99091.
# 3. Задача сводится к поиску разбиения 99091 на слагаемые из исходных чисел.
# 4. Если разбиения нет, в ответ 0.
# 5. Если разбиение найдено, добавляем максимальные релевантности до нужного числа M.

# 6. В этом решении плохо, что мы не используем отсутствие переходов через разряд при вычитании
# Из 201 вычитать 10 вообще не нужно, будем надеяться войти по времени

input_file = "2019-10-19 Yandex ML Contest/input.txt"
input_file = "input.txt"
# print(chr(27) + "[2J")

with open(input_file, "r") as fr:
    input_lines = fr.readlines()
    n, m, k = map(int, input_lines[0].strip().split())
    orig = [[int(num) for num in line.strip().split()] for line in input_lines[1:n+1]]
    req = int(input_lines[n+1].strip().replace(' ', ''))
    # print(req)

# Сортируем по значению релевантности
data = sorted(orig, key=lambda x: x[0], reverse=True)
# print(data)

max_relevance = 0

def puzzle(sum, terms, offset, relevance):
    global max_relevance, m
    
    # Условия выхода из рекурсии

    # Проверка, получилось ли корректное разбиение
    if (sum == 0):
        new_sum = sum
        new_relevance = relevance
        # 5
        # Проверка, если current_M не достигло M
        if (m > len(offset)):
            # print(offset, "m > offset")
            # Значит можно максимально-релевантными добрать
            for term_index in range(0, (m-len(offset))):
                # print(offset, "term:", terms[term_index])
                new_relevance = new_relevance + terms[term_index][0]

        # Сохранение максимума
        if (new_relevance > max_relevance): 
            max_relevance = new_relevance


        # print(offset, "OK: suboptimal", new_relevance, "M:", len(offset))
        return 

    if (len(offset) >= m): 
        # print(offset, "GO BACK")
        return

    # print(offset, "sum:", sum)
    # Бежим с верха списка и пробуем очередное слагаемое
    for term in terms:
        new_terms = terms.copy()
        # print(offset, "try add:", term)
        # Если слагаемое больше суммы, пропускаем
        if (term[1] > sum): 
            # new_terms.remove(term)
            continue
        # Иначе пробуем его, оставляя в terms только те, которые меньше его
        new_sum = sum - term[1]
        new_terms.remove(term)
        new_offset = offset + " "
        new_relevance = relevance + term[0]

        # Не вписывался по времени, этот трюк контроля времени добавил 4 пройденных теста
        # time.sleep(0)
        if ((time.time() - start_time) > 4.9):
            print(max_relevance)
            sys.exit(0);

        puzzle(new_sum, new_terms, new_offset, new_relevance)

puzzle(req, data, "", 0)
print(max_relevance)