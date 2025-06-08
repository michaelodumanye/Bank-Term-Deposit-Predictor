import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report

def train_logistic_regression(X_train, y_train):
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train, y_train)
    return lr

def train_random_forest(X_train, y_train):
    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)
    return rf

def train_xgboost(X_train, y_train):
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    xgb.fit(X_train, y_train)
    return xgb

def main(X_train, y_train, X_test, y_test):
    # Train models
    lr_model = train_logistic_regression(X_train, y_train)
    rf_model = train_random_forest(X_train, y_train)
    xgb_model = train_xgboost(X_train, y_train)
    
    # Evaluate on test set
    for name, model in [('Logistic Regression', lr_model), ('Random Forest', rf_model), ('XGBoost', xgb_model)]:
        y_pred = model.predict(X_test)
        print(f"--- {name} Classification Report ---")
        print(classification_report(y_test, y_pred))
    
    # Save models
    joblib.dump(lr_model, 'models/logistic_regression.joblib')
    joblib.dump(rf_model, 'models/random_forest.joblib')
    joblib.dump(xgb_model, 'models/xgboost.joblib')


if __name__ == "__main__":
    import pandas as pd
    import numpy as np
    from src.data_prep import main as prep_main  # assuming your data prep script is named data_prep.py

    # Prepare data (returns balanced and scaled train/test splits)
    X_train_bal, X_test_scaled, y_train_bal, y_test = prep_main()
    
    main(X_train_bal, y_train_bal, X_test_scaled, y_test)
