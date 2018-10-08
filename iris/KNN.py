import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score
from sklearn.metrics import classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import constant

def NotScaleKNN():
    classifier = KNeighborsClassifier(n_neighbors=9)
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    global avgNotScaleACC, avgNotScaleF1
    avgNotScaleACC += accuracy_score(y_pred, y_test)
    a = f1_score(y_pred, y_test, average=None)
    avgNotScaleF1 += np.mean(a)


def ScaleKNN():
    scaler = MinMaxScaler((-1,1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    classifier = KNeighborsClassifier(n_neighbors=9)
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)
    global avgScaleACC, avgScaleF1
    avgScaleACC += accuracy_score(y_pred, y_test)
    a =f1_score(y_pred, y_test, average=None)
    avgScaleF1 += np.mean(a)

if __name__ == "__main__":
    dataTrain = pd.read_csv('../dataset/Iris/irisTrain.csv')
    dataTest = pd.read_csv('../dataset/Iris/irisTest.csv')
    fileScale = open(constant.Iris_Scale_KNN, "w")
    fileNotScale = open(constant.Iris_Not_Scale_KNN, "w")
    X_train = dataTrain.iloc[:, :-1].values  # tach data
    y_train = dataTrain.iloc[:, -1].values  # tach nhan
    X_test = dataTest.iloc[:, :-1].values
    y_test = dataTest.iloc[:, -1].values
    avgNotScaleACC = 0
    avgScaleACC = 0
    avgNotScaleF1 = 0
    avgScaleF1 = 0
    times = 5
    for i in range(times):
        NotScaleKNN()
        ScaleKNN()
    # print not scale
    fileNotScale.write("ACC Not scale: " + (avgNotScaleACC/times).__str__())
    fileNotScale.write("\nF1 Not scale: " + (avgNotScaleF1/times).__str__())
    # print scale
    fileScale.write("ACC scale: " + (avgScaleACC/times).__str__())
    fileScale.write("\nF1 scale: " + (avgScaleF1/times).__str__())
