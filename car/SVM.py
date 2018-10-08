import numpy as np
from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

import constant

def NotScaleSVM():
    classifier = SVC()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    global avgNotScaleACC, avgNotScaleF1
    avgNotScaleACC += accuracy_score(y_pred, y_test)
    avgNotScaleF1 +=f1_score(y_pred, y_test, average='weighted')


def ScaleSVM():
    scaler = MinMaxScaler((-1,1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    classifier = SVC()
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)
    global avgScaleACC, avgScaleF1
    avgScaleACC += accuracy_score(y_pred, y_test)
    avgScaleF1 += f1_score(y_pred, y_test, average='weighted')
    fileNotScale.write("------Report------\n" + classification_report(y_test, y_pred))


if __name__ == "__main__":
    dataTrain = pd.read_csv('../dataset/car/carTrain.csv')
    # create dataTrain
    le = preprocessing.LabelEncoder()
    dataTrain = dataTrain.apply(le.fit_transform)
    # create dataTest
    dataTest = pd.read_csv('../dataset/car/carTest.csv')
    dataTest = dataTest.apply(le.fit_transform)
    fileScale = open(constant.Car_Scale_SVM, "w")
    fileNotScale = open(constant.Car_Not_Scale_SVM, "w")
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
        NotScaleSVM()
        ScaleSVM()
    # print not scale
    fileNotScale.write("ACC Not scale: " + (avgNotScaleACC/times).__str__())
    fileNotScale.write("\nF1 Not scale: " + (avgNotScaleF1/times).__str__())
    # print scale
    fileScale.write("ACC scale: " + (avgScaleACC/times).__str__())
    fileScale.write("\nF1 scale: " + (avgScaleF1/times).__str__())
