yandex = False
input_file = "D:\\Container\\Python Projects\\2020-09-30 Yandex ML Contest\\input_A.txt"
if (yandex): input_file = "input.txt"

with open(input_file, "r") as fr:
    n = int(fr.readline())
    for i in range(0, n):
        a, b = map(int, fr.readline().strip().split())
        print(a, b)

output_file = "D:\\Container\\Python Projects\\2020-09-30 Yandex ML Contest\\output_A.txt"
if (yandex): output_file = "output.txt"

with open(output_file, "w") as fw:
    fw.write(str.format("{0:.6f}", a))
    fw.write(str.format("{0:.6f}", b))