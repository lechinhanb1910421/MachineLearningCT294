# Lab 3
# Author: Le Nguyen Chi Nhan
# Student Code: B1910421
# Course: CT294-04

# Tap du lieu winequality-white co 11 thuoc tinh
# Cot nhan la cot "quality"
# Gia tri cua cac nhan la 3,4,5,6,7,8 (bien lien tuc)

from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
import pandas as pd
data = pd.read_csv("winequality-white.csv", delimiter=";")

X = data.iloc[:, :-1]
y = data.iloc[:, -1]

kHold_k = 50
KF = KFold(n_splits=kHold_k, shuffle=True, random_state=None)

clf_entropy = DecisionTreeClassifier(
    criterion="entropy", random_state=100, max_depth=3, min_samples_leaf=5)

acc_scores_DecTree = []
for train_index, test_index in KF.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    clf_entropy.fit(X_train, y_train)
    y_pred_clf = clf_entropy.predict(X_test)
    acc_clf = accuracy_score(y_pred_clf, y_test)*100
    acc_scores_DecTree.append(round(acc_clf, 3))

print("Values of accuracy score of Decision Tree:")
print(acc_scores_DecTree)
print()
print("Avarge accuracy score of Decision Tree: ",
      sum(acc_scores_DecTree)/kHold_k)
print()

# Values of accuracy score of Decision Tree:
# [53.061, 55.102, 59.184, 53.061, 51.02, 54.082, 48.98, 48.98, 47.959, 54.082,
#  39.796, 43.878, 53.061, 51.02, 44.898, 54.082, 51.02, 50.0, 45.918, 44.898,
# 44.898, 50.0, 48.98, 44.898, 61.224, 47.959, 54.082, 55.102, 53.061, 56.122,
# 51.02, 47.959, 53.061, 47.959, 53.061, 47.959, 52.041, 51.02, 47.959, 45.918,
# 43.878, 51.02, 47.959, 47.959, 54.082, 52.041, 50.0, 41.837, 53.608, 49.485]
# Do chinh xac trung binh sau 50 lan lap cua giai thuat Decision Tree la 50.20468


# Su dung giai thuat Bayes ngay tho va KNN de so sanh voi K=60
kHold_k = 60
KF = KFold(n_splits=kHold_k, shuffle=True, random_state=27)
KNN_Model = KNeighborsClassifier(n_neighbors=5)

Bayes_model = GaussianNB()

acc_scores_KNN = []
acc_scores_Bayes = []

for train_index, test_index in KF.split(X):
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]

    KNN_Model.fit(X_train, y_train)
    Bayes_model.fit(X_train, y_train)

    y_pred_KNN = clf_entropy.predict(X_test)
    y_pred_Bayes = Bayes_model.predict(X_test)

    acc_KNN = accuracy_score(y_pred_KNN, y_test)*100
    acc_Bayes = accuracy_score(y_pred_Bayes, y_test)*100

    acc_scores_KNN.append(round(acc_KNN, 3))
    acc_scores_Bayes.append(round(acc_Bayes, 3))

print("Avarge accuracy score of Bayes", sum(acc_scores_Bayes)/kHold_k)
print("Values of accuracy score of Bayes:")
print(acc_scores_Bayes)
print()
print("Avarge accuracy score of KNN", sum(acc_scores_KNN)/kHold_k)
print("Values of accuracy score of KNN:")
print(acc_scores_KNN)

# Do chinh xac trung binh cua giai thuat KNN: 52.57105
# Gia tri cua cac lan lap KNN
# [56.098, 47.561, 50.0, 64.634, 57.317, 50.0, 48.78, 57.317, 41.463, 56.098,
# 52.439, 57.317, 54.878, 50.0, 56.098, 53.659, 60.976, 56.098, 51.22, 53.659,
# 45.122, 53.659, 59.756, 46.341, 58.537, 50.0, 41.463, 60.976, 52.439, 46.341,
# 56.098, 45.122, 42.683, 54.878, 56.098, 51.22, 58.537, 50.0, 51.852, 45.679,
# 48.148, 56.79, 53.086, 45.679, 46.914, 53.086, 48.148, 53.086, 53.086, 49.383,
# 48.148, 51.852, 56.79, 48.148, 53.086, 56.79, 64.198, 48.148, 55.556, 61.728]

# Do chinh xac trung binh cua giai thuat Bayes Ngay Tho: 44.48625
# Gia tri cua cac lan lap Bayes Ngay Tho
# [35.366, 48.78, 48.78, 43.902, 50.0, 47.561, 46.341, 41.463, 46.341, 45.122,
# 51.22, 40.244, 47.561, 48.78, 45.122, 41.463, 43.902, 53.659, 37.805, 50.0,
# 37.805, 42.683, 47.561, 51.22, 52.439, 41.463, 45.122, 43.902, 39.024, 32.927,
# 40.244, 47.561, 40.244, 48.78, 48.78, 45.122, 35.366, 42.683, 45.679, 43.21,
# 48.148, 46.914, 37.037, 40.741, 35.802, 50.617, 43.21, 50.617, 44.444, 37.037,
# 41.975, 35.802, 56.79, 44.444, 41.975, 56.79, 48.148, 40.741, 37.037, 45.679]

# Bai tap 2: Xac dinh nam hay nu
data = pd.read_csv("exercise2.csv", delimiter=";")

data_x = data.iloc[:, 0:-1]
data_y = data.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(
    data_x, data_y, test_size=1/3.0, random_state=5)

clf_entropy = DecisionTreeClassifier(
    criterion="entropy", random_state=5, max_depth=3)

clf_entropy.fit(X_train, y_train)

y_pred = clf_entropy.predict(X_test)

print("Accuracy of Decision Tree using Entropy is : ",
      accuracy_score(y_test, y_pred)*100)

print("Predict with the given sample: ", clf_entropy.predict([[135, 39, 1]]))
# Ket qua du doan la 1
