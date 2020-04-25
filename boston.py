from sklearn.datasets import load_boston
boston = load_boston()

import pandas as pd
bos =pd.DataFrame(boston.data)
bos["PRICE"]= boston.target

X = bos.drop("PRICE",axis=1).values
Y = bos["PRICE"].values

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,Y)
y_pred = reg.predict(X_test)


from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,y_pred)