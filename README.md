# Bank Term Deposit Predictor

A machine learning solution to predict whether a client will subscribe to a bank term deposit, based on historical data from direct marketing campaigns conducted by a Portuguese banking institution.

📊 Overview
This end-to-end project includes data preprocessing, model training, and deployment via a user-friendly web application using Streamlit. It is structured to support experimentation and reproducibility.

## 📂 Project Structure
```
Bank-Term-Deposit-Predictor/
├── data/
│   └── raw/                      # Raw dataset (bank-additional-full.csv)
├── models/
│   └── bank_term_deposit_pipeline.joblib
├── notebooks/
│   ├── 01_eda.ipynb              # Exploratory Data Analysis
│   └── 03_modeling.ipynb         # Model training and evaluation
├── reports/
│   └── summary.pdf
├── src/
│   ├── data_prep.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
├── streamlit_app.py              # Streamlit interface
├── requirements.txt
├── .gitignore
└── README.md

```
🛠 Tools & Technologies
Python 3.10

Libraries: pandas, scikit-learn, joblib, streamlit, seaborn, matplotlib, xgboost, imbalanced-learn

Deployment: Streamlit Community Cloud

```

## 📈 Workflow
Perform exploratory data analysis (EDA)

Clean column names and encode categorical variables

Build and train a RandomForestClassifier inside a pipeline

Serialize the pipeline with joblib

Create a web app using Streamlit

Deploy and test the solution

```

## 📋 Dataset
Source: UCI Machine Learning Repository

Records: 41,188

Features: 20

Target: y (yes/no – did the client subscribe to a term deposit?)

```

## 🚀 How to Run
# Step 1: Create and activate a virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate

# Step 2: Install dependencies
pip install -r requirements.txt

# Step 3: Train model
python src/train_model.py

# Step 4: Launch the Streamlit app
streamlit run streamlit_app.py

```
## 🌐 Deployment
The app was deployed using Streamlit Community Cloud.

Note on Compatibility:
Model loading errors (e.g., Can't get attribute '_RemainderColsList') were resolved by aligning package versions:

streamlit==1.35.0
pandas==1.5.3
scikit-learn==1.2.2
joblib==1.2.0
numpy
matplotlib
seaborn
imbalanced-learn
xgboost

```

## ✅ Results
The trained pipeline loads and performs well in both local and deployed environments.

The Streamlit interface allows users to input data and receive real-time predictions.

The entire ML lifecycle is covered—from raw data to deployed product.

```
## 🔮 Future Improvements
Integrate SHAP or LIME for model interpretability

Add evaluation metrics to the app UI

Perform hyperparameter tuning

Add logging and monitoring

```

## 🧪 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

```

##✍️ Author
Michael Odumanye



