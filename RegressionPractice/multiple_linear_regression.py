import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Extracting the data
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Splitting the data into test data and training data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# Training the model
regressor = LinearRegression()
regressor.fit(X_train, Y_train)

# Predicting test results
y_pred = regressor.predict(X_test)

# print(y_pred.reshape(len(y_pred), 1), Y_test.reshape(len(Y_test), 1))

# Evaluating the R2 value of the model
score = r2_score(Y_test, y_pred)

print(score)  # 0.9325315554761303
