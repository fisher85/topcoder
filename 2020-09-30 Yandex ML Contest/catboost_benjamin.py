input_file = "D:\\Container\\Python Projects\\2020-09-30 Yandex ML Contest\\benjamin_dataset.json"

import pandas as pd
import numpy as np

# https://catboost.ai/docs/installation/python-installation-test-catboost.html#python-installation-test-catboost
from catboost import CatBoostClassifier

df = pd.read_json(input_file, orient='records')
y = df['label'].values
X = df.drop(columns=['label'])

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = CatBoostClassifier(iterations=10000, od_type="Iter", learning_rate=0.5).fit(X_train, y_train)
y_pred = model.predict(X_test)

import sklearn.metrics as metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score

print("Confusion matrix:\n", confusion_matrix(y_test, y_pred))

accuracy = metrics.accuracy_score(y_test, y_pred)
precision = metrics.precision_score(y_test, y_pred)
recall = metrics.recall_score(y_test, y_pred)
f1 = metrics.f1_score(y_test, y_pred)
auc_score = metrics.roc_auc_score(y_test, y_pred)

print('Accuracy =', accuracy)
print('Precision =', precision)
print('Recall =', recall)
print('F1 =', f1)
print('AUC ROC =', auc_score)

features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

for index, i in enumerate(indices[:100]):
    print('{}.\t#{}\t{:.6f}\t{}'.format(index + 1, i, importances[i], features[i]))