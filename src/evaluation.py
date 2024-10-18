from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

def evaluate_model(y_test, y_pred):
    confusion = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    return confusion, report, roc_auc
