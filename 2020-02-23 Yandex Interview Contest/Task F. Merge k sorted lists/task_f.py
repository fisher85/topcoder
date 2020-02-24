a = []
with open("c:/topcoder/topcoder/2020-02-23 Yandex Interview Contest/Task F. Merge k sorted lists/input.txt", "r") as fr:
    input_lines = fr.readlines()
    k = int(input_lines[0])
    for i in range(0, k):
        next_line = input_lines[i + 1].strip()
        num = next_line[0]
        b = [int(item) for item in next_line.split(' ')]
        b.pop(0)
        a.append(b)


