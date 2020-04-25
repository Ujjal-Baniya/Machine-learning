import pandas as pd
dataset = pd.read_csv("student-mat.csv",sep=';')
dataset2 = pd.read_csv("student-por.csv",sep=';')
dataset3 = dataset.append(dataset2)


X = dataset3.iloc[:,:30].values
Y = dataset3.iloc[:,30].values

from sklearn.preprocessing import LabelEncoder,OneHotEncoder
encoder = LabelEncoder()
X[:,0]=encoder.fit_transform(X[:,0])
X[:,1]=encoder.fit_transform(X[:,1])
X[:,3]=encoder.fit_transform(X[:,3])
X[:,4]=encoder.fit_transform(X[:,4])
X[:,5]=encoder.fit_transform(X[:,5])
X[:,8]=encoder.fit_transform(X[:,8])
X[:,9]=encoder.transform(X[:,9])
X[:,10]= encoder.fit_transform(X[:,10])
X[:,11]= encoder.fit_transform(X[:,11])
X[:,15]= encoder.fit_transform(X[:,15])
i=16
while i < 23:
    X[:,i]= encoder.transform(X[:,i])
    i+=1
onehotencoder_x=OneHotEncoder(categorical_features=[0])
X=onehotencoder_x.fit_transform(X).toarray()

from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.2,random_state=0)


from sklearn.linear_model import LinearRegression
reg = LinearRegression()
reg.fit(X,Y)
Y_pred = reg.predict(X_test)
Y_pred = Y_pred.astype(int)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test,Y_pred)