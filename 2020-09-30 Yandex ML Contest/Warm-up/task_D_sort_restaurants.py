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
# file_name = "c:\\topcoder\\topcoder\\2020-09-30 Yandex ML Contest\\Warm-up\\restaurants_train_only_d.txt"

df = pd.read_csv(file_name, delimiter="\t")
# print(df.head())

# Смотрим данные, есть повторы
# 1.0,-1.0,-1.0,0.01465590205,0.007281080354000001 - 3 раза
# 0.5,-1.0,-1.0,0.005612432957,0.0059392936529999996 - 2 раза
# 1.0,-1.0,5.525968655,0.01832646877,0.018520560119999998 - 2 раза
# 0.0,-1.0,-1.0,0.003585138591,0.004979998805 - 2 раза и др.
# print(df.value_counts())
# df.value_counts().to_frame().reset_index().to_csv("c:\\topcoder\\topcoder\\2020-09-30 Yandex ML Contest\\Warm-up\\restaurants_unique.txt")

# print(df["r1"].value_counts())
# print(df["r2"].value_counts())
# print(df["d1"].value_counts())
# print(df["d2"].value_counts())

"""
with open(file_name, "r", encoding="utf8") as fr:
    input_lines = fr.readlines()
    winner, r1, r2, d1, d2 = map(float, input_lines[0].strip().split("\t"))

print(winner, r1, r2, d1, d2)
"""

# Решение частного случая, когда для каждого ресторана задан только один фактор,
# пусть это будет расстояние. Тогда монотонной функцией скоринга будет линейная функция
# y = ax + b
# При этом каждая пара, записанная в виде неравенства:
# 0 4 2
# a*4 + b > a*2 > b
# 4a > 2a
# 2a > 0
# a > 0
# Всегда выполняется для любого a > 0 и любого b

# НО! Поскольку у нас метрика должна стремиться к 0.6, чтобы получить максимум баллов, нужно:
# 0.6 = log (1 + exp(delta))
# 3.981071706 = 1 + exp(delta)
# ln(2.981071706) = ln(exp(delta))
# delta = 1.092282869

# Т.е. лучший коэффициент из области определения a > 0 определяется метрикой

def score(r, d, coefs = [1, 1, 0]):
    result = coefs[0] * r + coefs[1] * d + coefs[2]
    return result

def get_metrics(df, coefs = [1, 1, 0]):
    epsilon = 0.0001
    n = 0
    m = 0 

    for index, row in df.iterrows():
        if (abs(row['winner'] - 0.5) < epsilon): continue 
        if (abs(row['r1'] - -1) < epsilon): continue 
        if (abs(row['r2'] - -1) < epsilon): continue 
        
        if (row['winner'] - 0 < epsilon):
            score_winner = score(row['r1'], row['d1'], coefs)
            score_looser = score(row['r2'], row['d2'], coefs)
        else:
            score_winner = score(row['r2'], row['d2'], coefs)
            score_looser = score(row['r1'], row['d1'], coefs)
        
        n = n + 1
        m = m + math.log(1 + math.exp(score_looser - score_winner))
        
        """
        print("#", index, "D1:", row['d1'], "D2:", row['d2'], "<>:", 
            row['winner'], "W:", score_winner, "L:", score_looser, 
            "Delta:", (score_looser - score_winner), 
            "Exp:", math.exp(score_looser - score_winner), 
            "Log:", math.log(1 + math.exp(score_looser - score_winner)),
            "N:", n, "M:", m)
        """
    
    m = m / n
    return m

file_name = "c:\\topcoder\\topcoder\\2020-09-30 Yandex ML Contest\\Warm-up\\restaurants.in"
with open(file_name, "r", encoding="utf8") as fr:
    n = int(fr.readline().strip())
    for i in range(0, n):
        d = float(fr.readline().strip())
        # print(score(d))

# Перебор коэффициентов
import numpy as np

steps = 10
bestCoefs = [0, 0, 0]
bestM = 9999999999

for c1 in np.linspace(1, 2, steps):
    for c2 in np.linspace(1, 2, steps):
        coefs = [c1, c2, 0]
        metrics = get_metrics(df, coefs)
        # print("A:", a, "Metrics:", metrics)
        if (metrics > 0.6 and metrics < bestM):
            bestCoefs = coefs
            bestM = metrics

print("Optimal Coefs:", bestCoefs, "Optimal Metrics:", bestM)