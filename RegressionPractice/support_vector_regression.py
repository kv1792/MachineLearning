import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.metrics import r2_score

# Reading the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

Y = Y.reshape(len(Y), 1)

# Splitting training set and test set
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# Feature Scaling
sc_x = StandardScaler()
sc_y = StandardScaler()
X_train = sc_x.fit_transform(X_train)
Y_train = sc_y.fit_transform(Y_train)

# Training the model
regressor = SVR(kernel='rbf')
regressor.fit(X_train, Y_train)

# Predicting the dependent variable - Y with test set X_test
Y_pred = sc_y.inverse_transform(regressor.predict(sc_x.transform(X_test)))

# Evaluating the model
score = r2_score(Y_test, Y_pred)

print(score)  # 0.9480784049986258
