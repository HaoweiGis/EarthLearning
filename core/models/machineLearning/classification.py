import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import classification_report,confusion_matrix
from sklearn import model_selection


def get_randomforest(input, testsize=0.2, n_estimators=1000, **kwargs):
    df= pd.read_csv(input,header=None) 
    X = np.array(df.iloc[:,1:])
    Y = np.array(df.iloc[:,[0]]).ravel()
    x,y = sklearn.utils.shuffle(X, Y, random_state=1)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=testsize, random_state=42)

    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators)
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)
    reports = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
    print(confusion)
    return model, reports 

def get_supportvector(input, testsize=0.2, **kwargs):
    df= pd.read_csv(input,header=None) 
    X = np.array(df.iloc[:,1:])
    Y = np.array(df.iloc[:,[0]]).ravel()
    x,y = sklearn.utils.shuffle(X, Y, random_state=1)
    x_train, x_test, y_train, y_test = model_selection.train_test_split(x, y, test_size=testsize, random_state=42)

    from sklearn import svm
    model = svm.SVC()
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)
    reports = classification_report(y_test, y_pred)
    confusion = confusion_matrix(y_test, y_pred, labels=None, sample_weight=None)
    print(confusion)
    return model, reports