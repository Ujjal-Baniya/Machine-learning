import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.naive_bayes import BernoulliNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

dataset = pd.read_csv("train.csv").drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
dataset = dataset.drop([61,829],axis=0)
Y = dataset['Survived']
X = dataset.drop('Survived',axis=1)

from sklearn.preprocessing import LabelEncoder
encoder_x = LabelEncoder()
X['Sex'] = encoder_x.fit_transform(X['Sex'])
X['Embarked'] = encoder_x.fit_transform(X["Embarked"])
X['Age'].fillna(value=X['Age'].mean(),inplace=True)

X_test = pd.read_csv("test.csv")
df = pd.DataFrame()
df['PassengerID'] = X_test.PassengerId
X_test = pd.read_csv("test.csv").drop(['PassengerId','Name','Ticket','Cabin'],axis=1)
encoder_xt = LabelEncoder()
X_test['Sex'] = encoder_xt.fit_transform(X_test['Sex'])
X_test['Embarked'] = encoder_xt.fit_transform(X_test["Embarked"])
X_test['Age'].fillna(value=X_test['Age'].mean(),inplace=True)
X_test['Fare'].fillna(value=X['Fare'].mean(),inplace=True)


NB = BernoulliNB()
NB.fit(X,Y)
Y_pred = NB.predict(X_test)
df['Survived']  = Y_pred
df.to_csv(r"C:\Users\UJJAL BANIYA\Downloads\result.csv",index = False, header=True)

result=pd.read_csv("result.csv")

