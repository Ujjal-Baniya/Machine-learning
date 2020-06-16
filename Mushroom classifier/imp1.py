# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 19:39:02 2020

@author: UJJAL BANIYA
"""

import pandas as pd
import numpy as np

dataset = pd.read_csv("mushroom.csv")
#Adding new attribute Symptoms                                     `
symp=[]
for x in dataset["class"]:
    if x=="e":
        symp.append("n")
    else:
        symp.append("y")
newDataset = dataset.assign(symptoms=symp)
symp.clear()

newDataset = dataset

#Assigning vairables
Y = newDataset["class"]
X = newDataset.iloc[:,1:]
  
#Label Encoding
from sklearn.preprocessing import LabelEncoder
encoder_y = LabelEncoder()
Y = encoder_y.fit_transform(Y)

#Checking and dropping columns containing NAN values temporarily
for items in X.columns:
     if X[items].isna().any():
         temp = items
         X.drop(items,axis=1,inplace=True)
         
encoder_x = LabelEncoder()
for items in X.columns:
    X[items]  = encoder_x.fit_transform(X[items])
    
    

#Handling nan values of stalk-root
stalk = newDataset[temp]
stalk.fillna("null",inplace=True)
s = stalk.unique()
i=0
for each in s:
    if each == "null" :
        continue
    else:
        stalk.replace(each,i,inplace=True)
        i+=1
stalk.replace("null",np.nan,inplace=True)
stalk.fillna(stalk.mean(),inplace=True)

#Adding new filtered column back to the X
X[temp]=stalk


#Dividing Data to training set and testing set
from sklearn.model_selection import train_test_split
train_x,test_x,train_y,test_y = train_test_split(X,Y,test_size=0.2,random_state=0)



from sklearn.linear_model import LinearRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


linear = LinearRegression()
linear.fit(train_x,train_y)
y1_pred = linear.predict(test_x)

logistic = LogisticRegression()
logistic.fit(train_x,train_y)
y2_pred = logistic.predict(test_x)

# =============================================================================
# gussian = GaussianNB()
# gussian.fit(train_x,train_y)
# y3_pred = gussian.predict(test_y)
# =============================================================================


print(accuracy_score(test_y,y1_pred))
