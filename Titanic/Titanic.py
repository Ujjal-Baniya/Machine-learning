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
X['Age'].fillna(value=X['Age'].median(),inplace=True)

onehotencoder_x=OneHotEncoder()
X=onehotencoder_x.fit_transform(X).toarray()

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.3,random_state=0)


#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

KNN = KNeighborsClassifier(n_neighbors = 7)
LR = LogisticRegression()
NB = BernoulliNB()

models = {'knn':KNN, 'lr': LR, 'nb':NB}

def train(model, X, Y):
    model.fit(X, Y)
        
def predict(model, X_test):
    return model.predict(X_test)

def accuracy(Y_pred, Y_test):
    return accuracy_score(Y_test, Y_pred)

pred = []
for k,v in models.items():
    train(v, X_train, Y_train)
    pred.append((k, accuracy(predict(v, X_test), Y_test)))
    


