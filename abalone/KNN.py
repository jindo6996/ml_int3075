import numpy as np
from sklearn import preprocessing
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
    avgNotScaleF1 += f1_score(y_pred, y_test, average='weighted')
    # fileNotScale.write("------Report------\n" + classification_report(y_test, y_pred))


def ScaleKNN():
    scaler = MinMaxScaler((-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    classifier = KNeighborsClassifier(n_neighbors=9)
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)
    global avgScaleACC, avgScaleF1
    avgScaleACC += accuracy_score(y_pred, y_test)
    avgScaleF1 += f1_score(y_pred, y_test, average='weighted')
    # fileScale.write("------Report------\n" + classification_report(y_test, y_pred))

if __name__ == "__main__":
    dataTrain = pd.read_csv('../dataset/abalone/abaloneTrain.csv')
    # create dataTrain
    le = preprocessing.LabelEncoder()
    dataTrain = dataTrain.apply(le.fit_transform)
    # create dataTest
    dataTest = pd.read_csv('../dataset/abalone/abaloneTest.csv')
    dataTest = dataTest.apply(le.fit_transform)
    fileScale = open(constant.Abalone_Scale_KNN, "w")
    fileNotScale = open(constant.Abalone_Not_Scale_KNN, "w")
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
