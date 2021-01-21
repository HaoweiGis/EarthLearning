import numpy as np
import pandas as pd
import sklearn
from sklearn import metrics

def get_randomforestRegr(input, testsize=0.2, max_depth=2, **kwargs):
    df= pd.read_csv(input,header=None) 
    X = np.array(df.iloc[:,1:])
    Y = np.array(df.iloc[:,[0]]).ravel()
    x,y = sklearn.utils.shuffle(X, Y, random_state=1)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=testsize, random_state=42)

    from sklearn.ensemble import RandomForestRegressor
    model = RandomForestRegressor(max_depth=max_depth)
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)
    reports = regression_report(y_test, y_pred)
    plotAnalysis = np.column_stack((y_test, y_pred))
    np.savetxt("plotAnalysis1.csv", plotAnalysis, delimiter=',')
    return model, reports


def get_supportvectorRegr(input, testsize=0.2, C=1.0, epsilon=0.2, **kwargs):
    df= pd.read_csv(input,header=None) 
    X = np.array(df.iloc[:,1:])
    Y = np.array(df.iloc[:,[0]]).ravel()
    x,y = sklearn.utils.shuffle(X, Y, random_state=1)
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=testsize, random_state=42)

    from sklearn.svm import SVR
    model = SVR(C=C , epsilon=epsilon)
    model.fit(x_train,y_train)

    y_pred = model.predict(x_test)
    reports = regression_report(y_test, y_pred)
    return model, reports


def regression_report(y_true, y_pred):
    explained_scores = metrics.explained_variance_score(y_true, y_pred, multioutput='raw_values')
    mean_absolute_errors = metrics.mean_absolute_error(y_true, y_pred, multioutput='raw_values')
    r2_scores = metrics.r2_score(y_true, y_pred, multioutput='raw_values')
    
    max_error = metrics.max_error(y_true, y_pred)
    explained_scosre = metrics.explained_variance_score(y_true, y_pred)
    mean_absolute_error = metrics.mean_absolute_error(y_true, y_pred)
    mean_squared_error = metrics.mean_squared_error(y_true, y_pred)
    mean_squared_log_error = metrics.mean_squared_log_error(y_true, y_pred)
    median_absolute_error = metrics.median_absolute_error(y_true, y_pred)
    r2_score = metrics.r2_score(y_true, y_pred, multioutput='variance_weighted')
    
    accs = {'explained_scores':explained_scores,'mean_absolute_errors':mean_absolute_errors,
    'r2_scores':r2_scores,'max_error':max_error,'explained_scosre':explained_scosre,
    'mean_absolute_error':mean_absolute_error,'mean_squared_error':mean_squared_error,
    'mean_squared_log_error':mean_squared_log_error,
    'median_absolute_error':median_absolute_error,
    'r2_score':r2_score}

    keys = list(accs.keys())

    reports = None
    for key in keys:
        if reports is None:
            reports = key + ' value: ' + str(accs.get(key)) + '\n'
        else:
            reports = reports + key + ' value: ' + str(accs.get(key)) + '\n'
    reports ='\n'+ reports + 'detail URL is: https://scikit-learn.org/stable/modules/model_evaluation.html'
    return reports
        