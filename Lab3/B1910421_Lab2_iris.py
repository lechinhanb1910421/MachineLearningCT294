from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
iris_dt = load_iris()
iris_dt.data[1:5]
iris_dt.target[1:5]

X_train, X_test, y_train, y_test = train_test_split(
    iris_dt.data, iris_dt.target, test_size=1/3.0, random_state=5)


KNN_Model = KNeighborsClassifier(n_neighbors=5)
KNN_Model.fit(X_train, y_train)

y_pred = KNN_Model.predict(X_test)


print("Accuracy of Iris KNN is : ", accuracy_score(y_test, y_pred)*100)
print(confusion_matrix(y_test, y_pred, labels=[2, 0, 1]))

from sklearn.naive_bayes import GaussianNB

import pandas as pd

irisData = pd.read_csv("iris.data")
X = irisData.iloc[:,0:4]
y = irisData.iloc[:,-1]
# X_Bayes_train, X_Bayes_test, y_Bayes_train, y_Bayes_test = train_test_split(X, y, test_size=0.3, random_state=0)
# iris_dt = load_iris()
# iris_dt.data[1:5]
# iris_dt.target[1:5]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 0.3, random_state=2)
model = GaussianNB()
model.fit(X_train, y_train)

Bayes_actual = y_test
Bayes_pred = model.predict(X_test)

Bayes_conf_matrix = confusion_matrix(Bayes_actual, Bayes_pred)
print("Accuracy of Iris Bayes is : ", accuracy_score(y_test, Bayes_pred)*100)
print(Bayes_conf_matrix)