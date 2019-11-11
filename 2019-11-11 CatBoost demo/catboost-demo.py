import sys
print(sys.version)

# https://catboost.ai/docs/installation/python-installation-test-catboost.html#python-installation-test-catboost
import numpy 
from catboost import CatBoostRegressor

dataset = numpy.array([[1,4,5,6],[4,5,6,7],[30,40,50,60],[20,15,85,60]])
train_labels = [1.2,3.4,9.5,24.5]
model = CatBoostRegressor(learning_rate=1, depth=6, loss_function='RMSE')
fit_model = model.fit(dataset, train_labels)
print(fit_model.get_params())

# https://catboost.ai/docs/concepts/python-usages-examples.html
from catboost import CatBoostRegressor
# Initialize data

train_data = [[1, 4, 5, 6],[4, 5, 6, 7],[30, 40, 50, 60]]
train_labels = [10, 20, 30]

eval_data = [[2, 4, 6, 8],[1, 4, 50, 60]]

# Initialize CatBoostRegressor
model = CatBoostRegressor(iterations=10, learning_rate=1, depth=2)
# Fit model
model.fit(train_data, train_labels)
# Get predictions
preds = model.predict(eval_data)