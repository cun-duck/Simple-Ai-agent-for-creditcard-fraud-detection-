from sklearn.ensemble import RandomForestClassifier

def build_model():
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    return model

def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model

def predict(model, X_test):
    y_pred = model.predict(X_test)
    return y_pred
