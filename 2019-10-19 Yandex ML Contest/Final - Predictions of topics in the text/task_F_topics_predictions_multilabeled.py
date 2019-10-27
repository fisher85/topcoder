from skmultilearn.dataset import load_dataset
import time
start_time = time.time()

pre_path = "2019-10-19 Yandex ML Contest/Final - Predictions of topics in the text/"
input_file = pre_path + "train.tsv"
max_input_lines = 50000 # из 126325
max_output_lines = 100 # из ~30000
# input_file = "input.txt"

learn_it = True
model_version = 5

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
        y.append(labels)

        # Предобработка текста и уборка мусора
        # Преобразование регистра пропускаем, векторайзер сделает

print("Обработка TF-IDF...")

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
from sklearn.svm import SVC

minC = 10

import pickle

if (learn_it == True):
    # Версия 2
    """
    clf = SVC(kernel='linear', C=minC, random_state=241)
    clf.fit(X, y)
    """
    # Версия 3, делаем ресемплинг, выравниваем баланс
    # Версия 5, всё таки multi-label пробуем за час до окончания
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import MultiLabelBinarizer

    y_multilabel = MultiLabelBinarizer().fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y_multilabel, test_size=0.4, random_state=0)

    print("Train:", len(y_train))
    print("Test:", len(y_test))
    print("Overall:", len(y_multilabel))

    from sklearn.linear_model import SGDClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.multiclass import OneVsRestClassifier
    
    sgd = SGDClassifier(random_state=42) #loss='hinge', penalty='l2', alpha=1e-3, random_state=42, max_iter=6, tol=None)
    lr = LogisticRegression()
    clf = OneVsRestClassifier(lr)
    
    
    from skmultilearn.problem_transform import BinaryRelevance
    from sklearn.svm import SVC
    # clf = BinaryRelevance(classifier=SVC(kernel='linear', C=minC, random_state=241), require_dense=[False, True])

    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test[1])
    print(y_pred[0])    

    print("Время обучения модели: ", (time.time() - start_time))

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

        y_indexes = []
        ind = 0
        for item in y[0]:
            if (item == 1):
                y_indexes.append(str(ind))
            ind += 1

        if (len(y_indexes) < 1): y_indexes.append("999")
        res_string = ",".join(y_indexes)

        print(f"{index}\t{res_string}", file=fw)

print("Время вывода результатов: ", count, (time.time() - start_time))