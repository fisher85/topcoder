# Дорешивание, смотрим разбор:
# https://habr.com/ru/company/yandex/blog/477452/#academy

input_file = "2019-10-19 Yandex ML Contest/input_B_2.txt"
input_file = "2019-10-19 Yandex ML Contest/input.txt"
# input_file = "input.txt"

with open(input_file, "r") as fr:
    input_lines = fr.readlines()
    n, m, k = map(int, input_lines[0].strip().split())
    scores, attributes = [], []
    for i in range(1, n+1):
        score, flags = input_lines[i].split()
        scores.append(int(score))
        attributes.append(list(map(int, flags)))
    req = list(map(int, input_lines[n+1].strip().split()))
    # print(req)

# Сортируем по значению релевантности
# data = sorted(orig, key=lambda x: x[0], reverse=True)
# print(data)

best_score = 0
version = 4

# Решение на 10 баллов: если есть ровно один флажок, достаточно взять A1 публикаций 
# с этим флажком, имеющих наибольшую релевантность (если таких карточек меньше, чем A1, 
# то искомого набора не существует), а остальные (m – A1) добрать оставшимися 
# карточками с наилучшей релевантностью.

# Первые два теста будут с ошибочным ответом, 3-10 тесты пройдут, 10 баллов будет

if version == 1:
    # Для сортировки по релевантности объединим два списка в один и отсортируем по значинию первого столбца
    stacked = []
    for i in range(0, n):
        stacked.append([scores[i], attributes[i]])
    stacked.sort(key=lambda x: x[0], reverse=True);
    
    used_count = 0
    for i in range(0, n):
        if used_count == req[0]:
            # Значит, набрали А1 публикаций с флажком
            break
        if stacked[i][1][0] == 1:
            used_count += 1
            best_score += stacked[i][0]
            stacked[i][1][0] = 999 # Помечаем публикацию как выбранную
    
    if used_count < req[0]:
        # Ответа нет, если на первом проходе не нашлось А1 публикаций с нужным флажком
        best_score = 0
    else:
        # Остальные (m – A1) добираем публикациями с наилучшей релевантностью
        for i in range(0, n):
            if used_count == m:
                break
            if stacked[i][1][0] != 999:
                used_count += 1
                best_score += stacked[i][0]

# Решение на 30 баллов: если m не превосходит 3, то можно найти ответ полным 
# перебором всевозможных O(n3) троек карточек, выбрать наилучший по суммарной 
# релевантности вариант, удовлетворяющий ограничениям.

if version == 2:
    for i1 in range(0, n):
        for i2 in range(0, n):
            for i3 in range(0, n):
                indexes = [i1, i2, i3]
                # Проверка на уникальность индексов
                if m == 2:
                    if i1 == i2:
                        continue
                if m == 3:
                    if i1 == i2 or i2 == i3 or i1 == i3:
                        continue

                # Проверяем, удовлетворяет ли ограничению выбор публикаций под номерами i1, i2, i3
                new_score = 0
                new_req = req.copy()
                for i in range(0, m):
                    new_score += scores[indexes[i]]
                    for ii in range(0, k):
                        new_req[ii] = new_req[ii] - attributes[indexes[i]][ii]
                req_ok = True
                for i in range(0, k):
                    if new_req[i] > 0:
                        req_ok = False
                        break
                # Если ограничение выполнено, проверяем, не достигли ли максимума best_score
                if req_ok == True:
                    if new_score > best_score:
                        best_score = new_score

# Решение на 70 баллов (на 50 баллов всё то же самое, только проще реализовать): 
# если есть не более 3 флажков, то можно разбить все публикации на 8 непересекающихся 
# групп по набору флажков, которыми они обладают: 000, 001, 010, 011, 100, 101, 110, 111. 
# Публикации в каждой группе нужно отсортировать по убыванию релевантности. 
# Далее можно за O(m4) перебрать, сколько лучших публикаций мы берём из групп 
# 111, 011, 110 и 101. Из каждой берём от 0 до m публикаций, в сумме не более m. 
# После этого становится ясно, сколько публикаций необходимо добрать из групп 100, 010 и 001, 
# чтобы требования удовлетворялись. Остаётся добрать до m оставшимися карточками 
# с наилучшей релевантностью.

def get_stacked(attribute_list, scores, attributes):
    result = []
    for i in range(0, len(scores)):
        if attributes[i] == attribute_list:
            result.append([scores[i], attributes[i]])
    result.sort(key=lambda x: x[0], reverse=True)
    return result

def add_publications(stacked_xxx, count, req, score, m):
    global input_file
    new_req = req.copy()
    new_score = score
    new_m = m + count
    for i in range(0, count):
        for j in range(0, 3):
            new_req[j] = new_req[j] - stacked_xxx[i][1][j]
        new_score += stacked_xxx[i][0]
        if input_file != "input.txt": print(stacked_xxx[i])
    return new_req, new_score, new_m

def get_super_stacked(stacked_xxx, xxx, super_stacked):
    result = super_stacked.copy()
    for i in range(xxx, len(stacked_xxx)):
        result.append([stacked_xxx[i][0], stacked_xxx[i][1]])
    return result

if version == 3:
    # Дополняем списки до трех элементов
    if k < 3:
        for j in range(0, 3-k):
            req.append(0)
            for i in range(0, n):
                attributes[i].append(0)

    # Формируем списки
    stacked_111 = get_stacked([1, 1, 1], scores, attributes)
    stacked_011 = get_stacked([0, 1, 1], scores, attributes)
    stacked_110 = get_stacked([1, 1, 0], scores, attributes)
    stacked_101 = get_stacked([1, 0, 1], scores, attributes)
    
    stacked_001 = get_stacked([0, 0, 1], scores, attributes)
    stacked_010 = get_stacked([0, 1, 0], scores, attributes)
    stacked_100 = get_stacked([1, 0, 0], scores, attributes)
    stacked_000 = get_stacked([0, 0, 0], scores, attributes)
    
    # Перебор по группам 111, 011, 110 и 101
    for i111 in range(0, m+1): # Еще одна поздно обнаруженная ошибка: не до M, а до M+1!!!
        for i011 in range(0, m+1):
            # if i111 + i011 > m: continue
            for i110 in range(0, m+1):
                # if i111 + i011 + i110 > m: continue
                for i101 in range(0, m+1):
                    if i111 + i011 + i110 + i101 > m: continue
                    if i111 > len(stacked_111): continue
                    if i011 > len(stacked_011): continue
                    if i110 > len(stacked_110): continue
                    if i101 > len(stacked_101): continue

                    # В этом месте разбиение нас устраивает, i111 + i011 + i110 + i101 <= m
                    if input_file != "input.txt": print(i111, i011, i110, i101)

                    new_score = 0
                    new_req = req.copy()
                    new_m = 0
                    
                    # Берем лучшие публикации
                    new_req, new_score, new_m = add_publications(stacked_111, i111, new_req, new_score, new_m)
                    new_req, new_score, new_m = add_publications(stacked_011, i011, new_req, new_score, new_m)
                    new_req, new_score, new_m = add_publications(stacked_110, i110, new_req, new_score, new_m)
                    new_req, new_score, new_m = add_publications(stacked_101, i101, new_req, new_score, new_m)

                    # Добираем публикации из групп 100, 010 и 001
                    i100 = i010 = i001 = 0
                    if new_req[0] > 0:
                        if new_req[0] > len(stacked_100): continue
                        i100 = new_req[0]
                        new_req, new_score, new_m = add_publications(stacked_100, i100, new_req, new_score, new_m)
                    if new_req[1] > 0:
                        if new_req[1] > len(stacked_010): continue
                        i010 = new_req[1]
                        new_req, new_score, new_m = add_publications(stacked_010, i010, new_req, new_score, new_m)
                    if new_req[2] > 0:
                        if new_req[2] > len(stacked_001): continue
                        i001 = new_req[2]
                        new_req, new_score, new_m = add_publications(stacked_001, i001, new_req, new_score, new_m)
                    
                    if input_file != "input.txt": print(new_req)

                    # Проверяем, удовлетворяет ли ограничению выбор публикаций
                    # Если дошли сюда, значит во всех разрядах требования выполнены
                    req_ok = True
                    if new_m > m: req_ok = False

                    # Тест 7
                    # if new_m < m and m - new_m > len(stacked_000):
                    #    req_ok = False

                    # Ошибка в 7 тесте была в том, что я добирал в конце только из 000, а нужно
                    # добирать публикациями с максимальными релевантностями со всех групп

                    # Собираем в super_stacked оставшиеся публикации
                    i000 = 0
                    super_stacked = []
                    super_stacked = get_super_stacked(stacked_000, i000, super_stacked)
                    super_stacked = get_super_stacked(stacked_001, i001, super_stacked)
                    super_stacked = get_super_stacked(stacked_011, i011, super_stacked)
                    super_stacked = get_super_stacked(stacked_010, i010, super_stacked)
                    super_stacked = get_super_stacked(stacked_100, i100, super_stacked)
                    super_stacked = get_super_stacked(stacked_101, i101, super_stacked)
                    super_stacked = get_super_stacked(stacked_110, i110, super_stacked)
                    super_stacked = get_super_stacked(stacked_111, i111, super_stacked)
                    super_stacked.sort(key=lambda x: x[0], reverse=True)

                    if new_m < m and m - new_m <= len(super_stacked):
                        new_req, new_score, new_m = add_publications(super_stacked, m - new_m, new_req, new_score, new_m)

                    if new_m != m: req_ok = False
                    # Если ограничение выполнено, добираем до M из группы 000 и проверяем, 
                    # не достигли ли максимума best_score
                    if req_ok == True:
                        if new_score > best_score:
                            best_score = new_score

                    if input_file != "input.txt": print(new_req, new_score, new_m)

# Полное решение: рассмотрим функцию динамического программирования dp[i][a]...[z]. 
# Это максимальная суммарная оценка релевантности, которую можно получить, использовав ровно i публикаций, 
# чтобы было ровно a публикаций с флажком A, ..., z публикаций с флажком Z. 
# Тогда исходно dp[0][0]...[0] = 0, а для всех остальных наборов параметров значение положим равным -1, 
# чтобы в дальнейшем максимизировать это значение. 
# Далее будем «вводить в игру» карточки по одной и с помощью каждой карточки улучшать значения этой функции: 
# для каждого состояния динамики (i, a, b, ..., z) с помощью j-й публикации с флажками (aj, bj, ..., zj) 
# можно попробовать совершить переход в состояние (i + 1, a + aj, b + bj, ..., z + zj) и проверить, 
# не улучшится ли результат в этом состоянии. 
# 
# Важно: при переходе нас не интересуют состояния, где i ≥ m, поэтому всего состояний у такой динамики не 
# более mk + 1, и итоговая асимптотика — O(nmk + 1). Когда посчитаны состояния динамики, ответом является состояние, 
# которое удовлетворяет ограничениям и даёт наибольшую суммарную оценку релевантности.
# 
# С точки зрения реализации для ускорения работы программы полезно хранить состояние динамического программирования 
# и флажки каждой из публикаций в упакованном виде в одном целом числе (см. код), 
# а не в списке или кортеже. Такое решение использует меньше памяти и позволяет эффективно производить 
# апдейты состояний динамики.

# Упаковка набора флажков в одно целое число, последний байт - num_items
def pack_state(num_items, counts):
    result = 0
    for count in counts:
        result = (result << 8) + count
    return (result << 8) + num_items

def get_num_items(state):
    return state & 255 # взять последний байт, там в упаковке num_items

def get_flags_counts(state, num_flags):
    flags_counts = [0] * num_flags
    state >>= 8
    for i in range(num_flags):
        flags_counts[num_flags - i - 1] = state & 255
        state >>= 8
    return flags_counts

if version == 4:
    dp = {0 : 0}
    for i in range(n):
        score = scores[i]
        state_delta = pack_state(1, attributes[i])
        dp_temp = {}
        for state, value in dp.items():
            # Важно: при переходе нас не интересуют состояния, где i ≥ m
            if get_num_items(state) >= m:
                continue
            new_state = state + state_delta
            if value + score > dp.get(new_state, -1):
                dp_temp[new_state] = value + score
        dp.update(dp_temp)

    best_score = 0
    for state, value in dp.items():
        if get_num_items(state) != m:
            continue

        flags_counts = get_flags_counts(state, k)
        satisfied_bounds = True
        for i in range(k):
            if flags_counts[i] < req[i]:
                satisfied_bounds = False
                break

        if not satisfied_bounds:
            continue

        if value > best_score:
            best_score = value

print(best_score)