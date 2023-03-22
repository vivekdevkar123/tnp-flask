import pandas as pd
import numpy as np
import pickle


data=pd.read_excel("AI Sheet.xlsx")
data.head()
y=data["Placed Status"]
y.head()
x=data.drop("Placed Status",axis=1)
x.head()
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
x_train.shape,y_train.shape,x_test.shape,y_test.shape
from sklearn.neighbors import KNeighborsClassifier
model=KNeighborsClassifier(n_neighbors=3)
model
sv=model.fit(x_train,y_train)
# model.score(x_test,y_test)
# model.predict([[8.70,9.0,8.4,10.0,8.5,10]])[0]

# X = np.array(df.iloc[:, 0:4])
# y = np.array(df.iloc[:, 4:])

# from sklearn.preprocessing import LabelEncoder
# le = LabelEncoder()
# y = le.fit_transform(y.reshape(-1))

# from sklearn.model_selection import train_test_split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# from sklearn.svm import SVC
# sv = SVC(kernel='linear').fit(X_train,y_train)

# file = "sheet.pkl"
# fileobj = open(file,'wb')
# pickle.dump()
pickle.dump(sv, open("AI_Sheet.pkl", 'wb'))


