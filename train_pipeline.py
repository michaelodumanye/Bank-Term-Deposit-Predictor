import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier

# Load your training data
data = pd.read_csv("data/raw/bank-additional-full.csv")  # update this path if needed

# Split into X and y
X = data.drop("y", axis=1)
y = data["y"]

# Define categorical columns (as used before)
categorical_cols = ['job', 'marital', 'education', 'default', 'housing', 'loan',
                    'contact', 'month', 'day_of_week', 'poutcome']

# ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
    ],
    remainder='passthrough'  # all numerical features
)

# Build pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(class_weight='balanced', random_state=42))
])

# Fit the pipeline
pipeline.fit(X, y)

# Save to models/
joblib.dump(pipeline, "models/bank_term_deposit_pipeline.joblib")
print("✅ Pipeline saved to models/bank_term_deposit_pipeline.joblib")
