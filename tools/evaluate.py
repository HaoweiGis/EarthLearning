import numpy as np
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix,accuracy_score





csv_name = r'C:\Users\hp\Downloads\urbanecoImg_merge.csv'
csv_Path = r'C:\Users\hp\Downloads\urbanecoImg_merge'

def scoreFor(csvname):
    df= pd.read_csv(csvname) 
    y = np.array(df[['Validation']])
    y_pred = np.array(df[['classification']])
    reports = classification_report(y, y_pred)
    confusion = confusion_matrix(y, y_pred, labels=None, sample_weight=None)
    score = accuracy_score(y, y_pred)
    print(confusion)
    print(reports)
    return score


if __name__ == "__main__":
    dflist = []
    for i in range(1,15,1):
        print(i)
        csvname = csv_Path + str(i) + '.csv'

        score = scoreFor(csvname)
        print(csvname + ': ' + str(score))

        df= pd.read_csv(csvname)
        dflist.append(df)

    # result = pd.concat(dflist)
    # result.to_csv(r'C:\Users\hp\Downloads\urbanecoImg_merge.csv', index=False)

    # score = scoreFor(csv_name)
    # print(csv_name + ':' + str(score))

#     df= pd.read_csv(csv_Path) 
#     y = np.array(df[['Validation']])
#     y_pred = np.array(df[['classification']])
#     reports = classification_report(y, y_pred)
#     confusion = confusion_matrix(y, y_pred, labels=None, sample_weight=None)
#     score = accuracy_score(y, y_pred)
#     print(y)