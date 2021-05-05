import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score

# Extracting data
dataset = pd.read_csv('Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, -1].values

# Splitting training and test data
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.2, random_state=0)


# Training the Polynomial regressor

# Setting the feature X with degree 4
poly_reg = PolynomialFeatures(degree=4)
# Transforming the feature X of the training data as per the polynomial degree 4
X_poly = poly_reg.fit_transform(X_train)

regressor = LinearRegression()
regressor.fit(X_poly, Y_train)

# Predicting the value
y_pred = regressor.predict(poly_reg.transform(X_test))


# Evaluating the model
score = r2_score(Y_test, y_pred)

print(score)  # 0.945819334714723l̥7
