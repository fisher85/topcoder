# D. Рестораны
# C:\python38\python.exe 'c:\topcoder\topcoder\2020-09-30 Yandex ML Contest\Warm-up\task_D_sort_restaurants.py'

# Регулярно пользователи Яндекс.Карт выбирают подходящий для них ресторан по множеству критериев. 
# Для упрощения будут рассмотрены два фактора, влияющие на их выбор: расстояние до пользователя и рейтинг организации. 
# Имеется несколько тысяч попарных оценок от реальных пользователей, в каждой из которых 
# одна пара (расстояние, рейтинг) сравнивается с другой. 
# 
# Необходимо построить модель, монотонно зависящую от двух этих факторов, которая согласуется с наибольшей долей оценок.


# Компилятор Python + ML постоянно сбоил, кроме того, при некорректном ответе получал не 0, а PE.
# Проверить себя смог, когда в ответ выдавал просто r


# 1. Сначала реализовал быстрый перебор коэффицентов, нашел для функции z = ax + by ответ [0, 0].
# Такой ответ получает 0 баллов, поскольку метрика = 0,693
# 2. Развил перебор, но стало понятно, что функция похоже на монотонную, можно спуском не зная производной лучще искать
# 3. Сделал поэлементный спуск со сменой направления и длины шага. Нашел для функций
# z = ax + by + cxy и z = ax + by + cxy + dx^2
# несколько ответов, набрал 2.3 балла из 4
# Каждый шаг проверял корректность (это быстрее расчета метрики), а потом уже считал метрику. 
# В метрику добавил штраф за большую разницу, когда рестораны с ничьей при обучении.

# Задача - полиномиальная регрессия, нелинейная. Хотя, возможно, нужно было градиентный спуск реализовывать
# Похоже, что моим подходом, доразвитым до высокой автоматизации, с монтекарловским стартом начальной точки, 
# можно было найти хороший ответ для многочлена более высокого порядка

import pandas as pd
import math

print(math.exp(81.4908368384145))
file_name = "c:\\topcoder\\topcoder\\2020-09-30 Yandex ML Contest\\Warm-up\\restaurants_train.txt"
# file_name = "c:\\topcoder\\topcoder\\2020-09-30 Yandex ML Contest\\Warm-up\\restaurants_train_only_d.txt"
file_name = "D:\\Container\\Python Projects\\2020-09-30 Yandex ML Contest\\Warm-up\\restaurants_train.txt"
df = pd.read_csv(file_name, delimiter="\t")
# df["d1"] = df["d1"] * 10
# df["d2"] = df["d2"] * 10
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

# НО! Поскольку у нас метрика должна стремиться быть меньше 0.6, чтобы получить максимум баллов, нужно:
# 0.6 = ln (1 + exp(delta))
# 1.822118 = 1 + exp(delta)
# ln(0.822118) = ln(exp(delta))
# delta = -0,195870368346319560719684178215

# Т.е. лучший коэффициент из области определения a > 0 определяется метрикой

# Почему ноль баллов за m >= 0.69
# Если всё время возвращать ноль, то delta = 0, exp(delta) = 1, log(2) = ln(2) = 0,6931471805599453

def score(r, d, coefs):
    result = coefs[0] * r + coefs[1] * d + coefs[2] * r * d + coefs[3] * r * r + coefs[4] * d * d
    return result

def get_metrics(df, coefs):
    penalty = 99999
    epsilon = 0.0001
    max_eps = 0
    n = 0
    m = 0 

    for index, row in df.iterrows():
       
        # Первый ресторан выиграл
        if (row['winner'] - 0 < epsilon): 
            score_winner = score(row['r1'], row['d1'], coefs)
            score_looser = score(row['r2'], row['d2'], coefs)
            n = n + 1
            try:
                m = m + math.log(1 + math.exp(score_looser - score_winner))                
            except:
                metrics = 999999999
                return metrics

        # Второй ресторан выиграл            
        if (row['winner'] - -1 < epsilon):
            score_winner = score(row['r2'], row['d2'], coefs)
            score_looser = score(row['r1'], row['d1'], coefs)
            n = n + 1
            try:
                m = m + math.log(1 + math.exp(score_looser - score_winner))                
            except:
                metrics = 999999999
                return metrics

        # При ничье нужно штрафовать за большую дельту
        if (abs(row['winner'] - 0.5) < epsilon):
            score_winner = score(row['r1'], row['d1'], coefs)
            score_looser = score(row['r2'], row['d2'], coefs)
            delta = (score_looser - score_winner) * (score_looser - score_winner)
            if delta > max_eps:
                max_eps = delta
                # print(row['r1'], row['d1'], row['r2'], row['d2'], score_winner, score_looser, delta)
      
        """
        print("#", index, "D1:", row['d1'], "D2:", row['d2'], "<>:", 
            row['winner'], "W:", score_winner, "L:", score_looser, 
            "Delta:", (score_looser - score_winner), 
            "Exp:", math.exp(score_looser - score_winner), 
            "Log:", math.log(1 + math.exp(score_looser - score_winner)),
            "N:", n, "M:", m)
        """
    
    m = m / n
    return m # + 1000 * max_eps

# Проверка, является ли решение корректным
def is_correct(df, coefs):
    correct = True
    for index, row in df.iterrows():
        score1 = score(row['r1'], row['d1'], coefs)
        score2 = score(row['r2'], row['d2'], coefs)
        if (abs(score1 - score2) > 700): # exp не возьмется от такого большого числа
            correct = False
            break
        if row['r2'] >= row['r1'] and row['d2'] <= row['d1'] and score2 < score1:
            correct = False
            break
        if row['r1'] >= row['r2'] and row['d1'] <= row['d2'] and score1 < score2:
            correct = False
            break
    return correct

"""
# Перебор коэффициентов
import numpy as np

steps = 20
bestCoefs = [50, 50, 50, 50, 50]
bestM = 9999999999

for c1 in np.linspace(-10, 10, steps):
    bestCoefsLocal = [50, 50, 50, 50, 50]
    bestMLocal = 9999999999
    for c2 in np.linspace(-10, 10, steps):
        for c3 in np.linspace(-10, 10, steps):
            for c4 in np.linspace(-10, 10, steps):
                for c5 in np.linspace(-10, 10, steps):
                    coefs = [c1, c2, c3, c4, c5]
                    if not is_correct(df, coefs): continue
                    metrics = get_metrics(df, coefs)
                    # print("A:", a, "Metrics:", metrics)
                    if (metrics > 0.5 and metrics < bestMLocal):
                        bestCoefsLocal = coefs
                        bestMLocal = metrics
                    if (metrics > 0.5 and metrics < bestM):
                        bestCoefs = coefs
                        bestM = metrics
    print(c1, "Local Coefs:", bestCoefsLocal, "Local Metrics:", str.format("{0:.3f}", bestMLocal), "SubOptimal Coefs:", bestCoefs, "SubOptimal Metrics:", str.format("{0:.3f}", bestM))
    if bestMLocal < bestM:
        bestCoefs = bestCoefsLocal
        bestM = bestMLocal

print("Optimal Coefs:", bestCoefs, "Optimal Metrics:", bestM)
"""

def descent(df, coefs, learning_rate, opt_metrics, error, max_steps):
    
    correct_opt = 999
    correct_coefs = [0, 0, 0, 0, 0]
    opt_coefs = coefs.copy()
    delta = 9999999999
    for step in range(max_steps):

        print("STEP", step)
        # Перебираем все коэффициенты, и на каждой позиции пробуем спуститься на step*отрезок
        for coef_index in range(len(coefs)):
            while delta > error:
                next_step = learning_rate
                try_next_coefs = coefs.copy()
                try_next_coefs[coef_index] = try_next_coefs[coef_index] + next_step
                print("Next step by coef", coef_index, ":", try_next_coefs)
                if 1 == 1 or is_correct(df, try_next_coefs):
                    metrics = get_metrics(df, try_next_coefs)
                    print("  Metrics:", metrics)
                    if metrics < opt_metrics:
                        delta = opt_metrics - metrics
                        opt_metrics = metrics
                        opt_coefs = try_next_coefs.copy()
                        coefs = try_next_coefs.copy()
                        print("***** New Suboptimal Metrics:", opt_metrics)
                        if is_correct(df, try_next_coefs):
                            print("CORRECT:", try_next_coefs)
                            correct_coefs = try_next_coefs.copy()
                            correct_opt = opt_metrics
                    else:

                        break
                else:
                    break
        
        
        if delta < error:
            print("***** STOP. DELTA LIMIT:", delta, " STEP:", step)
            print("OPT: ", opt_metrics)
            print("OPT COEFS: ", opt_coefs)
            print("CORRECT COEFS: ", correct_coefs)
            print("CORRECT METRICS: ", correct_opt)
            # break
        

        # Меняем шаг и направление
        learning_rate = - (learning_rate * 0.9)

    print("OPT: ", opt_metrics)
    print("OPT COEFS: ", opt_coefs)
    print("CORRECT COEFS: ", correct_coefs)
    print("CORRECT METRICS: ", correct_opt)

# Начинаем поиск
left_bounds = [-10000, -10000, -10000, -10000, -10000]
left_bounds = [1, -10, 1, 1, 2]
left_bounds = [0, 5, 0.37922665736154315, 0.1, -2]
left_bounds = [-0.5246550374604081, -1.556808915219689, 0.3839841518118844, 0.08649148282327007, -2]
left_bounds = [-0.677555037460408, -0.5151489152196889, 0.31108415181188437, 0.08649148282327007, -1.6289032790000002]
left_bounds = [-0.657154364711046, -1.0282587457392252, 0.3233788955368467, 0.08649148282327007, -1.6289032790000002] # 3929
left_bounds = [0.6, -10, 0.4, 2, -2]

# 0.636 [0.12868700000001043, -3.60570000000003, 0.30492999999999837] 2.29527195132 баллов
# result = coefs[0] * r + coefs[1] * d + coefs[2] * r * d
# 0.635 [0.13168700000001043, -3.8892000000000673, 0.30492999999999837]
# 0.632 [0.14168700000001044, -6.859200000000063, 0.30492999999999837, 0]

# 0.631285 [0.15999999999999992, -9.433000000000003, 0.3, 2.800000000000001]
# result = coefs[0] * r + coefs[1] * d + coefs[2] * r * d + coefs[3] * d * d

coefs = left_bounds
opt_metrics = 9999999999
learning_rate = 0.1
error = 1e-5
max_steps = 1000

descent(df, coefs, learning_rate, opt_metrics, error, max_steps)