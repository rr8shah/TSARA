import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv('histogram_data/TrainData_Histogram_256stride_variance.csv')
X_train = data.iloc[:,:2]
y_train = data.iloc[:,2]
y_train
clf = LogisticRegression()
clf.fit(X_train,y_train)
pred = clf.predict(X_train)
print("Train Data image level prediction accuracy is "+ str(accuracy_score(pred,y_train)))


data_val = pd.read_csv('histogram_data/ValData_Histogram_256stride_variance.csv')
X_val = data_val.iloc[:,:2]
y_val = data_val.iloc[:,2]
pred_val = clf.predict(X_val)
print("Validation Data image level prediction accuracy is "+ str(accuracy_score(pred_val,y_val)))


data_test = pd.read_csv('histogram_data/TestData_Histogram_256stride_variance_srno.csv')
X_test = data_test.iloc[:,1:3]
y_test = data_test.iloc[:,3]
sr_no = data_test.iloc[:,0]
pred_test = clf.predict(X_test)
print("Validation Data image level prediction accuracy is "+ str(accuracy_score(pred_test, y_test)))


decision = clf.decision_function(X_test)

log = clf.predict_log_proba(X_test)

proba = clf.predict_proba(X_test)


index = []
for i in range(len(proba)):
    index.append(np.argmax(proba[i]))


maximum = []
for i in range(len(index)):
    maximum.append(proba[i][index[i]])

df = pd.DataFrame(data={"sr_no":sr_no, "target": y_test, "prediction": pred_test, 'prediction_probability':maximum})
df.to_csv("TestData_CompleteImage_PredictionProbability_variance.csv", sep=',',index=False)


from sklearn.metrics import accuracy_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve


acc = accuracy_score(pred_test, y_test)
print('Testing Accuracy: %f' % acc)
precision = precision_score(y_test,pred_test, average='micro')
print('Precision: %f' % precision)
recall = recall_score(y_test,pred_test, average='micro')
print('Recall: %f' % recall)
f1 = f1_score(y_test,pred_test, average='micro')
print('F1 score: %f' % f1)


fpr, tpr, thresholds = roc_curve(pred_test, y_test)
roc_auc = auc(fpr, tpr)

plt.clf()
plt.plot(fpr, tpr, color='darkorange', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic For patch Size 200 and parameters 16k')
plt.legend(loc="lower right")
plt.savefig('LogisticRegression_ROC.png')


prec, rec, thresholds = precision_recall_curve(pred_test, y_test)
area = auc(rec, prec)
print("Area Under Curve: %0.2f" % area)


plt.plot(rec, prec, label='Precision-Recall curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall For patch Size 200 and parameters 16k: AUC=%0.2f' % area)
plt.legend(loc="lower left")
plt.savefig('LogisticRegression_PR.png')


from sklearn.metrics import confusion_matrix


confusion_matrix(pred_test, y_test, labels = [0, 1])
