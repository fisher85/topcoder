# Task 1 from Yandex Sample Contest
# https://contest.yandex.ru/contest/3/enter/

fr = open("input.txt", "r")
line = fr.readline()
parts = line.split()
a = int(parts[0])
b = int(parts[1])
sum = a + b
result = str(sum)

fw = open("output.txt", "w")
fw.write(result)