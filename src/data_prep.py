import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from imblearn.over_sampling import SMOTE
import os

def load_data(path):
    return pd.read_csv(path, sep=';')

def clean_data(df):
    # Example: Handle missing values, fix inconsistencies
    df = df.replace('unknown', pd.NA)
    df = df.dropna()  # or use imputation if preferred
    return df

def encode_data(df):
    # Label encode the target
    le = LabelEncoder()
    df['y'] = le.fit_transform(df['y'])

    # One-hot encode categorical variables except target
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    if 'y' in categorical_cols:
        categorical_cols.remove('y')

    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    return df

def scale_features(X_train, X_test):
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled, scaler

def handle_imbalance(X, y):
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    return X_res, y_res

def main():
    # Get absolute path to the current file's directory (src/)
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Point to raw data CSV inside ../data/raw/
    raw_path = os.path.abspath(os.path.join(base_dir, '..', 'data', 'raw', 'bank-additional-full.csv'))
    processed_path = os.path.abspath(os.path.join(base_dir, '..', 'data', 'processed', 'bank_processed.csv'))

    df = load_data(raw_path)
    df_clean = clean_data(df)
    df_encoded = encode_data(df_clean)

    X = df_encoded.drop('y', axis=1)
    y = df_encoded['y']

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    # Handle imbalance on training data only
    X_train_resampled, y_train_resampled = handle_imbalance(X_train_scaled, y_train)

    # Combine processed train set for saving (optional)
    df_train_processed = pd.DataFrame(X_train_resampled, columns=X.columns)
    df_train_processed['y'] = y_train_resampled.values

    # Create processed dir if not exists and save file
    os.makedirs(os.path.dirname(processed_path), exist_ok=True)
    df_train_processed.to_csv(processed_path, index=False)

    print(f"Processed data saved to {processed_path}")

    return X_train_resampled, X_test_scaled, y_train_resampled, y_test

