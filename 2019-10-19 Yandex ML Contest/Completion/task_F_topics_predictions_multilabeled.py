# Дорешивание 01-12-2019
# Мой результат в финале - 37.88, 100 место. При этом вообще multilabel решение не получилось

# Первый подход. Сделать простых 100 классификаторов для каждого тега и прогнать все 100 классификаторов. 
# https://scikit-learn.org/0.21/modules/generated/sklearn.multiclass.OneVsRestClassifier.html#sklearn.multiclass.OneVsRestClassifier

# Тестовый пример multi-label classification 
# http://scikit.ml/tutorial.html

"""
from skmultilearn.dataset import load_dataset
X_train, y_train, feature_names, label_names = load_dataset('emotions', 'train')
X_test, y_test, _, _ = load_dataset('emotions', 'test')
print(feature_names[:10])
print(label_names)

from skmultilearn.problem_transform import BinaryRelevance
from sklearn.svm import SVC
clf = BinaryRelevance(
    classifier=SVC(),
    require_dense=[False, True]
)

clf.fit(X_train, y_train)
print(clf.classifiers_)

prediction = clf.predict(X_test)
print(prediction)

import sklearn.metrics as metrics
print(metrics.hamming_loss(y_test, prediction))
print(metrics.accuracy_score(y_test, prediction))
"""

# Собственно задача
import time
start_time = time.time()

final_pre_path = "2019-10-19 Yandex ML Contest/Final - Predictions of topics in the text/"
completion_pre_path = "2019-10-19 Yandex ML Contest/Completion/"
input_file = final_pre_path + "train.tsv"
max_input_lines = 3000 # из 126325
max_output_lines = 50000 # из ~30000

# Обучать модель? Или взять сохраненную модель
learn_it = True
model_version = 7

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
        # One-label classification: y.append(labels[0])
        # Multi-label classification: y.append(labels)
        y.append(labels[0])

        # Предобработка текста и уборка мусора
        # Преобразование регистра пропускаем, векторайзер сделает

print("Обработка TF-IDF...")

stop_words = []
with open(final_pre_path + "stopwords-ru.txt", "r", encoding="utf-8") as file:
    for line in file:
        line = line.strip() 
        stop_words.append(line)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(X)
words = vectorizer.get_feature_names()
print("Время векторизации:", len(y), (time.time() - start_time))
# Время векторизации:  29920 записей - 20.52900218963623 секунд, БЫСТРО

print("Обучение модели...")
start_time = time.time()
# print(words)
    
from sklearn.svm import SVC
minC = 10

import pickle

if (learn_it == True):
    # Версия 2
    clf = SVC(kernel='linear', C=minC, random_state=241)
    clf.fit(X, y)
    y_pred = clf.predict(X[1])
    print(y_pred[0])    

    print("Время обучения модели:", (time.time() - start_time))

    with open(f"{completion_pre_path}finalmodel{model_version}.pkl", 'wb') as f:
        pickle.dump(clf, f)

else:

    with open(f"{completion_pre_path}finalmodel{model_version}.pkl", 'rb') as f:
        clf = pickle.load(f)

# Вывод результатов
print("Вывод результатов...")
start_time = time.time()
# print(clf.predict(X[1]))

test_file = final_pre_path + "test.tsv"
output_file = completion_pre_path + f"submission-by-model-{model_version}.tsv"

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

        y_indexes = []
        y_indexes.append(str(y[0]))

        """
        y_indexes = []
        ind = 0
        for item in y:
            if (item == 1):
                y_indexes.append(str(ind))
            ind += 1
        """

        if (len(y_indexes) < 1): y_indexes.append("999")
        res_string = ",".join(y_indexes)

        print(f"{index}\t{res_string}", file=fw)

print("Время вывода результатов:", count, (time.time() - start_time))