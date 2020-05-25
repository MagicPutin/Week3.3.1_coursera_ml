import pandas as pd
import sklearn.metrics

# 1st task
classification = pd.read_csv('data/classification.csv')
tp, fp, fn, tn = 0, 0, 0, 0
for i in range(classification.shape[0]):
    true = classification.iloc[i]['true']
    pred = classification.iloc[i]['pred']
    if true == 1 and pred == 1:
        tp += 1
    elif true == 0 and pred == 0:
        tn += 1
    elif true == 1 and pred == 0:
        fn += 1
    elif true == 0 and pred == 1:
        fp += 1

with open('answers/task1.txt', 'w') as task1:
    task1.write(str(tp) + ' ' + str(fp) + ' ' + str(fn) + ' ' + str(tn))

# 2nd task
accuracy = (tp + tn)/classification.shape[0]
precision = tp/(tp + fp)
recall = tp/(tp + fn)
F_value = 2 * (precision * recall)/(precision + recall)

with open('answers/task2.txt', 'w') as task2:
    task2.write(str(round(accuracy, 2)) + ' ' + str(round(precision, 2)) + ' ' + str(round(recall, 2)) + ' ' +
                str(round(F_value, 2)))

# 3rd task
scores = pd.read_csv('data/scores.csv')
a_max = 0
method = ' '
for i in scores.columns[1:]:
    a = sklearn.metrics.roc_auc_score(scores['true'], scores[i])
    if a > a_max:
        a_max = a
        method = i

with open('answers/task3.txt', 'w') as task3:
    task3.write(method)

# 4th task
method = ' '
precision_max = 0
for i in scores.columns[1:]:
    precision, recall, thresholds = sklearn.metrics.precision_recall_curve(scores['true'], scores[i])
    for j in range(len(thresholds)):
        if (recall[j] >= 0.7) and (precision[j] >= precision_max):
            method = i
            precision_max = precision[j]


with open('answers/task4.txt', 'w') as task4:
    task4.write(method)







