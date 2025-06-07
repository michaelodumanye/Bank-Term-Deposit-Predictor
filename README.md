# Bank Term Deposit Predictor

A machine learning project to predict whether a client will subscribe to a bank term deposit based on features from direct marketing campaigns.

## 📂 Project Structure
```
Azubi Project/
├── data/
│   └── raw/
├── notebooks/
│   ├── 01_eda.ipynb
│   └── 03_modeling.ipynb
├── reports/
│   └── summary.pdf
├── src/
│   ├── data_prep.py
│   ├── feature_engineering.py
│   ├── train_model.py
│   └── evaluate_model.py
├── models/
│   └── random_forest_model.joblib
├── requirements.txt
├── .gitignore
└── README.md
```

## 📈 Workflow
1. Perform EDA
2. Clean and encode data
3. Engineer features
4. Train a classification model
5. Evaluate performance
6. Report insights and recommendations

## 📊 Dataset
- Source: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/ml/datasets/Bank+Marketing)
- File used: `bank-additional-full.csv`

## 🚀 How to Run
```bash
# Create environment
python -m venv venv
source venv/bin/activate  # On Windows use venv\Scripts\activate

# Install requirements
pip install -r requirements.txt

# Run training
python src/train_model.py
```

## 🧪 Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix
