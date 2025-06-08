import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

data = pd.read_csv("data/raw/bank-additional-full.csv", sep=';')
data.columns = data.columns.str.strip()  # Clean column names

X = data.drop("y", axis=1)
y = data["y"]

categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                    'contact', 'month', 'day_of_week', 'poutcome']

preprocessor = ColumnTransformer(
    transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)],
    remainder='passthrough'
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

pipeline.fit(X, y)

joblib.dump(pipeline, "models/bank_term_deposit_pipeline.joblib")

# Test loading right away
loaded_pipeline = joblib.load("models/bank_term_deposit_pipeline.joblib")
print("Pipeline loaded successfully!")

