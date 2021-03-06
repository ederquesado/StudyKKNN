import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_validate, KFold,StratifiedKFold , cross_val_predict, cross_val_score
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.metrics import classification_report, confusion_matrix, make_scorer, accuracy_score, precision_score, recall_score, f1_score
from scipy.io.arff import loadarff


def classification_report_with_accuracy_score(y_true, y_pred):

    print(classification_report(y_true, y_pred)) # print classification report
    return accuracy_score(y_true, y_pred) # return accuracy score


raw_data = loadarff('../content/kc1.arff')
df_data = pd.DataFrame(raw_data[0])

scaler = StandardScaler()
scaler.fit(df_data.drop('defects', axis = 1))
df_pattern = scaler.transform(df_data.drop('defects', axis = 1))
df_params = pd.DataFrame(df_pattern, columns=df_data.columns[:-1])

X=df_params
y=df_data['defects'].map({b'true':1 ,b'false':0})

n_splits = 10
kf = StratifiedKFold(n_splits=n_splits, shuffle=True)

error_rate = []
accuracy = []
precision = []
recall = []
f1_score_list = []

k = [1, 2, 3, 5, 7, 9, 11, 13, 15]
for value in k:
    knn_cros_val = KNeighborsClassifier(n_neighbors=value, metric='euclidean', weights='distance')
    knn_cros_val.fit(X, y)
    cros_val = cross_validate(knn_cros_val, X, y, cv=kf)
    cross_val_predicted = cross_val_predict(knn_cros_val, X, y, cv=kf)
    cross_score_accuracy = cross_val_score(knn_cros_val, X, y, cv=kf, scoring=make_scorer(accuracy_score))
    cross_score_precision = cross_val_score(knn_cros_val, X, y, cv=kf, scoring=make_scorer(precision_score))
    cross_score_recall = cross_val_score(knn_cros_val, X, y, cv=kf, scoring=make_scorer(recall_score))
    cross_score_f1_score = cross_val_score(knn_cros_val, X, y, cv=kf, scoring=make_scorer(f1_score))
    # cross_score = cross_val_score(knn_cv, X,y, cv=kf, scoring=make_scorer(classification_report_with_accuracy_score))
    conf_mat = confusion_matrix(y, cross_val_predicted)

    error_rate.append(np.mean(cross_val_predicted != y))
    accuracy.append(cross_score_accuracy.mean() * 100)
    precision.append(cross_score_precision.mean() * 100)
    recall.append(cross_score_recall.mean() * 100)
    f1_score_list.append(cross_score_f1_score.mean() * 100)

    print("Accuracy of Model with Cross Validation is:", cross_score_accuracy.mean() * 100)
    print("Precision of Model with Cross Validation is:", cross_score_precision.mean() * 100)
    print("Recall of Model with Cross Validation is:", cross_score_recall.mean() * 100)
    print("F1-Score of Model with Cross Validation is:", cross_score_f1_score.mean() * 100)
    print(conf_mat)
    print("*" * 50)

plt.figure(figsize=(14,8))
plt.plot(k,error_rate,color='blue',linestyle='dashed',marker='o')
plt.xlabel('K')
plt.ylabel('Taxa de erro')
plt.show()

plt.figure(figsize=(14,8))
plt.plot(k,accuracy,color='blue',linestyle='dashed',marker='o')
plt.xlabel('K')
plt.ylabel('Accuracy')
plt.show()

plt.figure(figsize=(14,8))
plt.plot(k,precision,color='blue',linestyle='dashed',marker='o')
plt.xlabel('K')
plt.ylabel('Precision')
plt.show()

plt.figure(figsize=(14,8))
plt.plot(k,recall,color='blue',linestyle='dashed',marker='o')
plt.xlabel('K')
plt.ylabel('Recall')
plt.show()

plt.figure(figsize=(14,8))
plt.plot(k,f1_score_list,color='blue',linestyle='dashed',marker='o')
plt.xlabel('K')
plt.ylabel('F1-Score')
plt.show()