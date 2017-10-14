# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 19:02:55 2017

@author: absol
"""

import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

data = pd.read_csv('Salary_Data.csv')

X = data.iloc[:, -1].values
Y = data.iloc[:, 0].values
X = X.reshape((30,1))
Y = Y.reshape((30,1))

from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 1/3, random_state = 0)

#No feature scaling required because library(linear_model) takes care of it
#Simple Linear Regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor = regressor.fit(X_train, Y_train)

#Predicting using the trained model
y_pred = regressor.predict(X_test)

#Plotting the actual vs. predicted data
