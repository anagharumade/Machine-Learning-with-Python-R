# -*- coding: utf-8 -*-
"""
Created on Sat Oct 14 20:05:49 2017

@author: absol
"""
import pandas as pd

#importing data
data = pd.read_csv('50_startups.csv')
X = data.iloc[:,:-1].values
Y = data.iloc[:,4].values

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(category_features = [4])
X = onehotencoder.fit_transform(X)

#Spliting the data into training and testing data
from sklearn.cross_validation import train_test_split
X_train, Y_train, X_test, Y_test = train_test_split(X, Y)

#Multiple linear regression
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()

