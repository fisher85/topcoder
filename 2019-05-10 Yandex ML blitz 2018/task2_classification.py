# Task 2 (classification) from Yandex ML Blitz 2018
# https://habr.com/ru/company/yandex/blog/359435/

# https://towardsdatascience.com/named-entity-recognition-and-classification-with-scikit-learn-f05372f07ba2
# https://www.kdnuggets.com/2017/03/email-spam-filtering-an-implementation-with-python-and-scikit-learn.html

# Доступный пример байесовского классификатора
# https://appliedmachinelearning.blog/2017/05/23/understanding-naive-bayes-classifier-from-scratch-python-code/

# https://github.com/fisher85/Coursera-ML-Yandex/blob/master/07%20%D0%90%D0%BD%D0%B0%D0%BB%D0%B8%D0%B7%20%D1%82%D0%B5%D0%BA%D1%81%D1%82%D0%BE%D0%B2.ipynb

# Загрузите объекты из новостного датасета 20 newsgroups
from sklearn import datasets
newsgroups = datasets.fetch_20newsgroups(subset='all', categories=['alt.atheism', 'sci.space'])
y = newsgroups.target
X = newsgroups.data

# Вычислите TF-IDF-признаки для всех текстов
# TF (term frequency) * IDF (inverse document frequency)
from sklearn.feature_extraction.text import TfidfVectorizer

v = TfidfVectorizer()
v.fit_transform({'hello, world','hello, happy new year'})
print(v.get_feature_names())

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(X)
words = vectorizer.get_feature_names()
print(words[10000:10005])