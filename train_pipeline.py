import pandas as pd
import joblib
from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Load the data
data = pd.read_csv("data/raw/bank-additional-full.csv", sep=';')
data.columns = data.columns.str.strip()  # Clean column names

# Separate features and target
X = data.drop("y", axis=1)
y = data["y"]

# Specify categorical columns
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                    'contact', 'month', 'day_of_week', 'poutcome']

# Build the transformer and pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # This creates the `_RemainderColsList` internally
)

pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Fit the pipeline
pipeline.fit(X, y)

# Ensure the models folder exists
Path("models").mkdir(parents=True, exist_ok=True)

# Save the pipeline using joblib (with correct version)
joblib.dump(pipeline, "models/bank_term_deposit_pipeline.joblib")

# Test: Load immediately to confirm it's working
loaded_pipeline = joblib.load("models/bank_term_deposit_pipeline.joblib")
print("✅ Pipeline saved and loaded successfully!")


