# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 00:38:12 2020

@author: UJJAL BANIYA
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv('train.csv').dropna()
X = dataset.iloc[:,:-1].values
Y = dataset.iloc[:,1].values

"""from sklearn.preprocessing import Imputer
imputer = Imputer"""""

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,Y_train)
plt.scatter(X_test,Y_test,color='r')
plt.plot(X_test,regressor.predict(X_test),'b')
plt.show()
print(regressor.coef_)
