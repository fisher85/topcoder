yandex = False
input_file = "D:\\Container\\Python Projects\\2020-09-30 Yandex ML Contest\\benjamin_dataset_small.json"
input_file = "D:\\Container\\Python Projects\\2020-09-30 Yandex ML Contest\\benjamin_dataset.json"
if (yandex): input_file = "input.txt"

import pandas as pd
import numpy as np

df = pd.read_json(input_file, orient='records')
# df.describe().to_csv("D:\\Container\\Python Projects\\2020-09-30 Yandex ML Contest\\describe.txt")
# print(df.describe())

total_rows = 15151
n_rows = 1000
df.drop(df.tail(total_rows - n_rows).index,inplace=True)

df_new = df.copy()
print(df_new.shape)

for column in df.columns:
    # print(column, df[column].count())
    if (df[column].std() < 0.0001):
        df_new.drop(columns=[column], inplace=True)
        continue    
    if (df[column].count() != n_rows):
        df_new.drop(columns=[column], inplace=True)

"""
df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
df_new = df.copy()
"""
print("DF_NEW shape:", df_new.shape)

y = df_new['label'].values
X = df_new.drop(columns=['label'])
print("X shape:", X.shape, " y shape:", y.shape)

from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X = ss.fit_transform(X)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

unique, counts = np.unique(y_train, return_counts=True)
print("unique, counts:")
print(dict(zip(unique, counts)))

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

"""
rf = RandomForestClassifier(n_estimators=500, max_features=None, random_state=42, oob_score=True)
rf = RandomForestClassifier(random_state=42, oob_score=True)
rf.fit(X_train, y_train)
print('R^2 Training Score: {:.2f} \nR^2 Validation Score: {:.2f} \nOut-of-bag Score: {:.2f}'
      .format(rf.score(X_train, y_train), rf.score(X_test, y_test), rf.oob_score_))

clf.fit(X_train, y_train)
features = X.columns
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

for index, i in enumerate(indices):
    print('{}.\t#{}\t{:.6f}\t{}'.format(index + 1, i, importances[i], features[i]))
"""

import time
import warnings
warnings.filterwarnings("ignore")

from sklearn.metrics import roc_auc_score
from sklearn import model_selection
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

models = []
# models.append(('KNN', KNeighborsClassifier()))
# models.append(('SVM', SVC(gamma='auto')))
models.append(('CART', DecisionTreeClassifier(max_depth=5)))
# models.append(('RF', RandomForestClassifier(n_estimators=500, max_features=None, random_state=42, oob_score=True)))    
#models.append(('ABoost', AdaBoostClassifier()))
#models.append(('LR', LogisticRegression(solver='lbfgs', max_iter=200)))
#models.append(('NB', GaussianNB()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('QDA', QuadraticDiscriminantAnalysis()))
#models.append(('MLP', MLPClassifier()))

print('Model\tAcc\tPr\tRecall\tF1\tAUC\tExecution')
      
for name, model in models:
    start_time = time.time()
    kfold = model_selection.KFold(n_splits=3, random_state=24)    

    accuracy = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy').mean()
    precision = cross_val_score(model, X_train, y_train, cv=kfold, scoring='precision').mean()
    recall = cross_val_score(model, X_train, y_train, cv=kfold, scoring='recall').mean()
    f1_score = cross_val_score(model, X, y, cv=kfold, scoring='f1_weighted').mean()
    auc_score = cross_val_score(model, X, y, cv=kfold, scoring='roc_auc').mean()
    
    delta = time.time() - start_time
    print('{}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.2f} secs'.format(name, accuracy, precision, recall, f1_score, auc_score, delta))


clf = LinearDiscriminantAnalysis(store_covariance=True).fit(X_train, y_train)
print(clf.get_params(deep=True))
print(X_train.head())
X2 = clf.transform(X)
print(X2.head())


"""
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
"""