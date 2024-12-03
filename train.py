import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import common
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

data = pd.read_csv("Data/data.csv")


X = data.drop(columns=['trip_duration'])
y = data['trip_duration']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=common.RANDOM_STATE)

model = Ridge()
model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)
score = mean_squared_error(y_test, y_pred)
print(f"Score on train data {score:.2f}")
common.persist_model(model, common.MODEL_PATH)