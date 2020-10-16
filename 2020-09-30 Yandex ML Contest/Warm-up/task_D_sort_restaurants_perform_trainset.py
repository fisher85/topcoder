# D. Рестораны
# C:\python38\python.exe 'c:\topcoder\topcoder\2020-09-30 Yandex ML Contest\Warm-up\task_D_sort_restaurants.py'

# Регулярно пользователи Яндекс.Карт выбирают подходящий для них ресторан по множеству критериев. 
# Для упрощения будут рассмотрены два фактора, влияющие на их выбор: расстояние до пользователя и рейтинг организации. 
# Имеется несколько тысяч попарных оценок от реальных пользователей, в каждой из которых 
# одна пара (расстояние, рейтинг) сравнивается с другой. 
# 
# Необходимо построить модель, монотонно зависящую от двух этих факторов, которая согласуется с наибольшей долей оценок.

import pandas as pd
import math

file_name = "c:\\topcoder\\topcoder\\2020-09-30 Yandex ML Contest\\Warm-up\\restaurants_train.txt"

import sys
sys.stdin = open(file_name, "r", encoding="utf-8")

with open(file_name, "r", encoding="utf8") as fr, open(file_out, "w") as fw:
    input_lines = fr.readlines()
    for line in input_lines[1:]:
        winner, r1, r2, d1, d2 = map(float, line.strip().split("\t"))
        # print(winner, r1, r2, d1, d2)
        fw.write(str.format("{0:.12f}\t", r1) + str.format("{0:.12f}\n", d1))
        fw.write(str.format("{0:.12f}\t", r2) + str.format("{0:.12f}\n", d2))

