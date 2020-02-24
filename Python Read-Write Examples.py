k = int(input())
distr = {}
for _ in range(k):
    next_line = input().split()
    for i in next_line[1:]:
        if i in distr:
            distr[i] += 1
        else:
            distr[i] = 1

for i in range(101):
    i = str(i)
    if i in distr:
        print(' '.join([i] * distr[i]), end=' ')

# Чтение из файла
fr = open("input.txt", "r")
line = fr.readline()
parts = line.split()

###
with open("2019-05-10 Yandex ML blitz 2018/input1_1.csv", "r") as fr:
      input_lines = fr.readlines()
      b = int(input_lines[0])
      X = [[int(num) for num in line.split(",")] for line in input_lines[1:]]

###
with open(file_name, "r") as fr:
    data_orig = [[float(num) for num in line.split()] for line in fr]

###
with open(file_input, "r") as fr, open(file_output, "w") as fw:
    input_lines = fr.readlines()
    n, m = map(int, input_lines[0].strip().split())
    
    # Первое значение всегда идет в output
    prev = int(input_lines[1])
    fw.write("0 ")

###
with open(input_file, "r") as fr:
    data_orig = [[float(num) for num in line.split()] for line in fr]

###
with open(input_file, "r") as fr:
    input_lines = fr.readlines()
    n, m, k = map(int, input_lines[0].strip().split())
    orig = [[int(num) for num in line.strip().split()] for line in input_lines[1:n+1]]
    req = int(input_lines[n+1].strip().replace(' ', ''))



# Запись в файл
fw = open("output.txt", "w")
fw.write(result)

###
output_name = "2019-05-12 Yandex ML blitz 2018 final/output_A_12.txt"
with open(output_name, "w") as fw:
    fw.write(str.format("{0:.6f}", a) + " ")
    fw.write(str.format("{0:.6f}", b) + " ")
    fw.write(str.format("{0:.6f}", c))

###
print(*result, sep=' ')

###
print(a[0,:])
print(format(np.sum(a[0,:]), ".8f"))
print(b[0])

###
with open(input_name, "r") as fr:
    test_data = [[float(num) for num in line.split()] for line in fr]
result = np.sum(test_data, axis=1)
np.savetxt(output_name, result, fmt='%0.8f')