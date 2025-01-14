from sklearn.metrics import (precision_score,
                             recall_score,
                             f1_score,
                             confusion_matrix)



def calculate_metrics(y_true, y_pred):
    if len(y_true) == 0 or len(y_pred) == 0:
        return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    accuracy = sum(y1 == y2 for y1, y2 in zip(y_true, y_pred)) / len(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.shape != (2, 2):
        tn, fp, fn, tp = 0, 0, 0, 0
        if len(cm) == 1:
            if y_true[0] == 1:
                tp = cm[0, 0]
            else:
                tn = cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()
    tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0

    return precision, recall, f1, tpr, fpr, fnr, accuracy
