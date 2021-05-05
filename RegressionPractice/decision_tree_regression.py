import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score

# Reading the dataset
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values


# Splitting training and test data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)

# Training the model
regressor = DecisionTreeRegressor(random_state=0)
regressor.fit(X_train, Y_train)

# Predicting the dependent variable - Y for the given X_test data
Y_pred = regressor.predict(X_test)

# Evaluating the model
score = r2_score(Y_test, Y_pred)

print(score)  # 0.9226091050550043
