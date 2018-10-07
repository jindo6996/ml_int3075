import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import time
import constant

def NotScaleSVM():
    file = open(constant.Iris_Not_Scale_SVM,"w")
    start_time = time.time()
    classifier = SVC()
    classifier.fit(X_train, y_train)
    y_pred = classifier.predict(X_test)
    end_time = time.time()
    file.write("------Confusion Matrix------\n"+np.array2string(confusion_matrix(y_test, y_pred), separator=', '))
    file.write("\n\n------Report------\n"+classification_report(y_test, y_pred))
    file.write('\n\n-------------\n* Accuracy is: '+ accuracy_score(y_pred, y_test).__str__())
    file.write("\n\n* Running time: %.2f (s)" % (end_time - start_time))

def ScaleSVM():
    file = open(constant.Iris_Scale_SVM,"w")
    scaler = MinMaxScaler((-1,1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    start_time = time.time()
    classifier = SVC()
    classifier.fit(X_train_scaled, y_train)
    y_pred = classifier.predict(X_test_scaled)
    end_time = time.time()
    file.write("------Confusion Matrix------\n"+np.array2string(confusion_matrix(y_test, y_pred), separator=', '))
    file.write("\n\n------Report------\n"+classification_report(y_test, y_pred))
    file.write('\n\n-------------\n* Accuracy is: '+ accuracy_score(y_pred, y_test).__str__())
    file.write("\n\n* Running time: %.2f (s)" % (end_time - start_time))
if __name__ == "__main__":
    dataTrain = pd.read_csv('../dataset/Iris/irisTrain.csv')
    dataTest = pd.read_csv('../dataset/Iris/irisTest.csv')
    X_train = dataTrain.iloc[:, :-1].values  # tach data
    y_train = dataTrain.iloc[:, -1].values  # tach nhan
    X_test = dataTest.iloc[:, :-1].values
    y_test = dataTest.iloc[:, -1].values
    NotScaleSVM()
    ScaleSVM()
