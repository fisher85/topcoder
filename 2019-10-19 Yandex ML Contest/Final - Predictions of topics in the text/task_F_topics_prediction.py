# Плохо, так и не доделал SVC в multi-label обертке
# Правильно советуют организаторы, если на простом решении получил немного, порадовался и хватит оттачивать
# Бери сразу сложный алгоритм и пытайся на нем baseline сделать
# Я до 10:00 второго дня настраивал и баловался с one-label, думая что за час сделаю усложнение
# ОШИБКА!!!

# https://contest.yandex.ru/contest/14696/problems/

# В Яндекс.Дзене много материалов, и чтобы пользователям было проще ориентироваться 
# в огромном многообразии контента, мы используем тематические теги. 
# При помощи алгоритма машинного обучения мы определяем, к каким темам относится определённый документ, 
# и показываем его подходящим пользователям. 
# В этом задании вам нужно построить такой алгоритм классификации документов.

# Multi-label classification!!!
# https://www.kaggle.com/roccoli/multi-label-classification-with-sklearn

# Доделка 3, пытаемся сделать multi-label classification
# Формально, для каждого тега нужен отдельных классификатор, хорошо что тегов 100
# http://scikit.ml/tutorial.html
from skmultilearn.dataset import load_dataset
import time
start_time = time.time()

# Чтение данных
print("Чтение данных...")

# В файле с обучающей выборкой в каждой строке через табуляцию записаны числовой id документа, 
# заголовок, содержимое, список числовых идентификаторов тем. 
# Все идентификаторы тем в списке — целые числа от 0 до 99, разделённые запятыми.
# Формат train.tsv:

# 0 Интересный заголовок Много букв в статье 1,2,3  
# 1 Классный заголовок Какой-то текст 3  
# ...

pre_path = "2019-10-19 Yandex ML Contest/Final - Predictions of topics in the text/"
input_file = pre_path + "train.tsv"
max_input_lines = 5000 # из 126325
max_output_lines = 50000 # из ~30000
# input_file = "input.txt"

learn_it = True
model_version = 4

# Выводы после второй попытки. Даже при max_input_lines = 10000 из 126325 все очень медленно, 
# решение готовится минут 10, записывается минут 10
# Значит, из обучающей выборки нужно выбрать 10000, но сбалансированно

X = []
y = []
count = 0
with open(input_file, "r", encoding="utf-8") as fr:
    for input_line in fr:
        count += 1
        if (count > max_input_lines): break

        splits = input_line.split('\t')
        # 1406 строка разбита переносом строки, проще пропустить
        if (len(splits) != 4): continue

        index = int(splits[0])
        topic = splits[1].strip()
        text = splits[2].strip()
        text_labels = splits[3].strip()

        labels = [int(num) for num in text_labels.split(',')] 
        # Текст = заголовок + текст
        X.append(topic + text)
        # Пока One-label classification
        y.append(labels[0])

        # Предобработка текста и уборка мусора
        # Преобразование регистра пропускаем, векторайзер сделает



# Базовую обработку текстов заимствуем из решенных заданий курса Coursera + Yandex + ВШЭ
# https://github.com/fisher85/Coursera-ML-Yandex/blob/master/07%20%D0%90%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7%20%D1%82%D0%B5%D0%BA%D1%81%D1%82%D0%BE%D0%B2.ipynb
print("Обработка TF-IDF...")

# Вычисляем TF-IDF-признаки для всех текстов
# TF (term frequency) * IDF (inverse document frequency)

# Подгружаем стоп-слова, возможно в векторайзере вместо этого поиграть параметром max_df
# If None, no stop words will be used. max_df can be set to a value in the range [0.7, 1.0) 
# to automatically detect and filter stop words based on intra corpus document frequency of terms.
stop_words = []
with open(pre_path + "stopwords-ru.txt", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip() 
        stop_words.append(line)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(X)
words = vectorizer.get_feature_names()
print("Время векторизации: ", len(y), (time.time() - start_time))
# Время векторизации:  29920 записей - 20.52900218963623 секунд, БЫСТРО
start_time = time.time()
# print(words)
    
# Подбор МИНИМАЛЬНОГО лучшего параметра C (penalty parameter C of the error term)
# Внимание, подбор выполняется долго!
import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

"""
# 10^6 не будет в сетке
grid = {'C': np.power(10.0, np.arange(-2, 3))}
cv = KFold(n_splits=5, shuffle=True, random_state=241)
clf = SVC(kernel='linear', random_state=241)
gs = GridSearchCV(clf, grid, scoring='accuracy', cv=cv)
gs.fit(X, y)
index = 0
for paramC in gs.cv_results_['params']:
    print(paramC, 'score =', gs.cv_results_['mean_test_score'][index])
    index += 1

# Два варианта: 1.0 и 10.0, долго подбираем на обучающей выборке длины 3000 строк
# Для начала используем 1.0, в последних попытках можно взять 10.0
"""
minC = 10

import pickle

if (learn_it == True):
    """
    # Подготовка к кросс-валидации, разделяем датасет на части
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MultiLabelBinarizer

    X_train_0, X_test_0, y_train_0, y_test_0 = train_test_split(X, y, test_size=0.4, random_state=0)
    y_multilabel = MultiLabelBinarizer().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_multilabel, test_size=0.4, random_state=0)

    print("Train:", len(y_train))
    print("Test:", len(y_test))
    print("Overall:", len(y_multilabel))
    
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    from sklearn.metrics import hamming_loss
    import numpy as np
    def hamming_score(y_true, y_pred, normalize=True, sample_weight=None):
        '''
        Compute the Hamming score (a.k.a. label-based accuracy) for the multi-label case
        http://stackoverflow.com/q/32239577/395857
        '''
        acc_list = []
        for i in range(y_true.shape[0]):
            set_true = set( np.where(y_true[i])[0] )
            set_pred = set( np.where(y_pred[i])[0] )
            tmp_a = None
            if len(set_true) == 0 and len(set_pred) == 0:
                tmp_a = 1
            else:
                tmp_a = len(set_true.intersection(set_pred))/float(len(set_true.union(set_pred)) )
            acc_list.append(tmp_a)
        return np.mean(acc_list)

    def print_score(y_pred, y_test, clf):
        print("Clf: ", clf.__class__.__name__)
        print("Hamming loss: {}".format(hamming_loss(y_pred, y_test)))
        print("Hamming score: {}".format(hamming_score(y_pred, y_test)))
        print("Clf score: {}".format(clf.score))
        print("---")  

    nb_clf = MultinomialNB()
    sgd = SGDClassifier(loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=6, tol=None)
    lr = LogisticRegression()
    mn = MultinomialNB()

    for classifier in [nb_clf, sgd, lr, mn]:
        clf = OneVsRestClassifier(classifier)
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        print_score(y_test, y_pred, classifier)

    # Обучение SVM по всей выборке с оптимальным параметром C, найденным на предыдущем шаге
    clf = SVC(kernel='linear', C=minC, random_state=241)
    y_train_0 = []
    for item in y_train:
        y_train_0.append(item[0])
    # В условии задачи указана метрика 
    # sklearn.metrics.f1_score(y_true, y_pred, average=’samples’)
    scores = cross_val_score(clf, X_train_0, y_train_0, cv=5)  #, scoring='f1_samples'
    print(scores)
    """
    
    # Версия 2
    """
    clf = SVC(kernel='linear', C=minC, random_state=241)
    clf.fit(X, y)
    """

    # Версия 3, делаем ресемплинг, выравниваем баланс, плохо получилось, выкидываем
    # Хотя не факт, на 5000 вместо 10000 без ресемплинга показал 34 против 38 очков
    # Если время будет, нужно запустить на 10000
    from imblearn.over_sampling import RandomOverSampler
    from sklearn.model_selection import train_test_split
    ros = RandomOverSampler(random_state=9000)
    X_resampled, y_resampled = ros.fit_sample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.4, random_state=9000)
    clf = SVC(kernel='linear', C=minC, random_state=241)
    clf.fit(X_train, y_train)

    # Версия 5, всё таки multi-label пробуем за час до окончания
    # Смотри файл _multilabeled.py

    print("Время обучения модели: ", (time.time() - start_time))

    # TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words=stop_words)
    # [0.26498423 0.31023102 0.30877193 0.36501901 0.33196721]
    # TfidfVectorizer()
    # [0.27444795 0.30693069 0.34385965 0.36121673 0.3442623 ]
    # TfidfVectorizer(stop_words=stop_words)
    # [0.26182965 0.30693069 0.32982456 0.36121673 0.3442623 ]
    # TfidfVectorizer(max_df=0.8, stop_words=stop_words)
    # [0.26182965 0.30693069 0.32982456 0.36121673 0.3442623 ]
    # TfidfVectorizer(max_df=0.5, stop_words=stop_words)
    # [0.26182965 0.30693069 0.32982456 0.36121673 0.3442623 ]
    # Не вижу разницы, возможно ошибся

    with open(f"{pre_path}finalmodel{model_version}.pkl", 'wb') as f:
        pickle.dump(clf, f)

else:

    with open(f"{pre_path}finalmodel{model_version}.pkl", 'rb') as f:
        clf = pickle.load(f)


# Вывод результатов
print("Вывод результатов...")
start_time = time.time()
# print(clf.predict(X[1]))

test_file = pre_path + "test.tsv"
output_file = pre_path + "submission.tsv"

count = 0
with open(test_file, "r", encoding="utf-8") as fr, open(output_file, "w") as fw:
    for input_line in fr:
        count += 1
        # Строк в файле test - ~30000, 6 минут на вывод
        # Индексы от 126048 до 157560, всего 31512
        if (count > max_output_lines): break
        splits = input_line.split('\t')
        # На тестовой выборке не пройдет, т.к. постпроцессор вернет ошибку 
        # Wrong set of document ids!
        # Поэтому исправляем датасет вручную
        # if (len(splits) != 3): continue 

        index = int(splits[0])
        topic = splits[1].strip()
        text = splits[2].strip()

        X = []
        X.append(topic + text)
        X = vectorizer.transform(X)
        y = clf.predict(X[0])

        print(f"{index}\t{y[0]}", file=fw)

print("Время вывода результатов: ", count, (time.time() - start_time))