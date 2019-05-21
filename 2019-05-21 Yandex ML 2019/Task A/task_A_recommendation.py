file_input = "2019-05-21 Yandex ML 2019/Task A/input.txt"
file_output = "2019-05-21 Yandex ML 2019/Task A/output.txt"
file_input = "input.txt"
file_output = "output.txt"

with open(file_input, "r") as fr, open(file_output, "w") as fw:
    input_lines = fr.readlines()
    n, m = map(int, input_lines[0].strip().split())
    
    # Первое значение всегда идет в output
    prev = int(input_lines[1])
    fw.write("0 ")

    # В очереди может храниться только одно значение, но много разных индексов
    from collections import deque
    q_value = 0
    q = deque()
        
    # Просмотр файла с третьей строки
    for i in range(1,n):
        b = int(input_lines[i+1])

        # Можно ли текущее значение сразу в вывод?
        if b != prev:
            fw.write("{0} ".format(i))
            prev = b
            # Сразу проверим, нельзя ли из очереди вывести значение
            if len(q) > 0 and q_value != b:
                fw.write("{0} ".format(q.popleft()))
                prev = q_value
        else:
            # Накапливаем в памяти текущее значение
            # Если очередь пустая, создаем ее
            if len(q) == 0:
                q_value = b
                q.append(i)
            else: # Иначе дописываем в конец запомненный индекс
                q.append(i)