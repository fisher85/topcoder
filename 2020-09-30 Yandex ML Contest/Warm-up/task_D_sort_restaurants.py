# D. Рестораны
# C:\python38\python.exe 'c:\topcoder\topcoder\2020-09-30 Yandex ML Contest\Warm-up\task_D_sort_restaurants.py'

# Регулярно пользователи Яндекс.Карт выбирают подходящий для них ресторан по множеству критериев. 
# Для упрощения будут рассмотрены два фактора, влияющие на их выбор: расстояние до пользователя и рейтинг организации. 
# Имеется несколько тысяч попарных оценок от реальных пользователей, в каждой из которых 
# одна пара (расстояние, рейтинг) сравнивается с другой. 
# 
# Необходимо построить модель, монотонно зависящую от двух этих факторов, которая согласуется с наибольшей долей оценок.

file_name = "c:\\topcoder\\topcoder\\2020-09-30 Yandex ML Contest\\Warm-up\\restaurants_train.txt"

import pandas as pd
df = pd.read_csv(file_name, delimiter="\t")
print(df.head())

# Смотрим данные, есть повторы
# 1.0,-1.0,-1.0,0.01465590205,0.007281080354000001 - 3 раза
# 0.5,-1.0,-1.0,0.005612432957,0.0059392936529999996 - 2 раза
# 1.0,-1.0,5.525968655,0.01832646877,0.018520560119999998 - 2 раза
# 0.0,-1.0,-1.0,0.003585138591,0.004979998805 - 2 раза и др.
print(df.value_counts())
df.value_counts().to_frame().reset_index().to_csv("c:\\topcoder\\topcoder\\2020-09-30 Yandex ML Contest\\Warm-up\\restaurants_unique.txt")

print(df["r1"].value_counts())
print(df["r2"].value_counts())
print(df["d1"].value_counts())
print(df["d2"].value_counts())

"""
with open(file_name, "r", encoding="utf8") as fr:
    input_lines = fr.readlines()
    winner, r1, r2, d1, d2 = map(float, input_lines[0].strip().split("\t"))

print(winner, r1, r2, d1, d2)
"""