# -*- coding: utf-8 -*-
"""
Created on Fri Oct 13 07:47:49 2017

@author: absol
"""

import pandas as pd

data = pd.read_csv('data.csv')
train = data.iloc[:,:-1].values
test = data.iloc[:,3].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values = 'NaN', strategy = 'mean', axis=0)
imputer = imputer.fit(train[:,1:3])
train[:,1:3] = imputer.transform(train[:,1:3])

#Encoding categorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
train[:,0] = labelencoder.fit_transform(train[:,0])
onehotencoder = OneHotEncoder(categorical_features = [0])
train = onehotencoder.fit_transform(train).toarray()
labelencoder_test = LabelEncoder()
test = labelencoder_test.fit_transform(test)