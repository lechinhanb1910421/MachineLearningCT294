from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
iris_dt = load_iris()
iris_dt.data[1:5]
iris_dt.target[1:5]

X_train, X_test, Y_train, Y_test = train_test_split(
    iris_dt.data, iris_dt.target, test_size=1/3.0, random_state=5)


KNN_Model = KNeighborsClassifier(n_neighbors=5)
KNN_Model.fit(X_train, Y_train)

y_pred = KNN_Model.predict(X_test)
KNN_Model.predict([[4, 4, 3, 3]])
print("Accuracy is ", accuracy_score(Y_test, y_pred)*100)

print(confusion_matrix(Y_test, y_pred, labels=[2, 0, 1]))
