from src.preprocessing import load_data, split_data, handle_imbalance
from src.model import build_model, train_model, predict
from src.evaluation import evaluate_model

data = load_data('data/creditcard.csv')
X_train, X_test, y_train, y_test = split_data(data)
X_train_smote, y_train_smote = handle_imbalance(X_train, y_train)

model = build_model()
model = train_model(model, X_train_smote, y_train_smote)

y_pred = predict(model, X_test)
confusion, report, roc_auc = evaluate_model(y_test, y_pred)

print(confusion)
print(report)
print(f"ROC AUC Score: {roc_auc:.2f}")
