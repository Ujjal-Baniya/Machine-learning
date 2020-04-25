import pandas as pd

dataset = pd.read_csv("mushrooms.csv")
Y = dataset["class"]
X = dataset.iloc[:,1:]


from sklearn.preprocessing import LabelEncoder
encoder_y = LabelEncoder()
Y = encoder_y.fit_transform(Y)

for col in X.columns:
    if X[col].nunique()==1:
        print(col)
        X.drop(col,axis=1,inplace=True)
    
encoder_x = LabelEncoder()
for col in X.columns:
    X[col]=encoder_x.fit_transform(X[col])
    
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
reg = LogisticRegression()
reg.fit(X_train,Y_train)
y_pred = reg.predict(X_test)

from sklearn.naive_bayes import BernoulliNB
classifier = BernoulliNB()
classifier.fit(X_train,Y_train)
y2_pred = reg.predict(X_test)

from sklearn.metrics import confusion_matrix
CM= confusion_matrix(Y_test,y_pred)
CMNB = confusion_matrix(Y_test,y2_pred)

print(CM)