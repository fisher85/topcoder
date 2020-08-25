# Правильные скобочные последовательности
# https://neerc.ifmo.ru/wiki/index.php?title=%D0%9F%D1%80%D0%B0%D0%B2%D0%B8%D0%BB%D1%8C%D0%BD%D1%8B%D0%B5_%D1%81%D0%BA%D0%BE%D0%B1%D0%BE%D1%87%D0%BD%D1%8B%D0%B5_%D0%BF%D0%BE%D1%81%D0%BB%D0%B5%D0%B4%D0%BE%D0%B2%D0%B0%D1%82%D0%B5%D0%BB%D1%8C%D0%BD%D0%BE%D1%81%D1%82%D0%B8
n = int(input())

def gen(n, open_count, close_count, str):
    if (open_count + close_count == 2*n):
        print(str)
        return
    if (open_count < n):
        gen(n, open_count + 1, close_count, str + "(")
    if (open_count > close_count):
        gen(n, open_count, close_count + 1, str + ")")

gen(n, 0, 0, "")