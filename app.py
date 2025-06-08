import streamlit as st
import pandas as pd
import joblib

# Load your trained model pipeline
pipeline = joblib.load("models/bank_term_deposit_pipeline.joblib")

st.title("Bank Term Deposit Prediction App")

st.markdown("### Please enter the customer's information below:")

# Categorical input mappings
job_options = {
    "Admin.": "admin.", "Blue-collar": "blue-collar", "Entrepreneur": "entrepreneur",
    "Housemaid": "housemaid", "Management": "management", "Retired": "retired",
    "Self-employed": "self-employed", "Services": "services", "Student": "student",
    "Technician": "technician", "Unemployed": "unemployed", "Unknown": "unknown"
}
marital_options = {"Single": "single", "Married": "married", "Divorced": "divorced", "Unknown": "unknown"}
education_options = {
    "Primary": "basic.4y", "Secondary": "basic.6y", "High School": "basic.9y",
    "Vocational": "professional.course", "University": "university.degree",
    "Illiterate": "illiterate", "Unknown": "unknown"
}
default_options = {"Yes": "yes", "No": "no", "Unknown": "unknown"}
housing_options = {"Yes": "yes", "No": "no", "Unknown": "unknown"}
loan_options = {"Yes": "yes", "No": "no", "Unknown": "unknown"}
contact_options = {"Cellular": "cellular", "Telephone": "telephone", "Unknown": "unknown"}
month_options = {
    "January": "jan", "February": "feb", "March": "mar", "April": "apr",
    "May": "may", "June": "jun", "July": "jul", "August": "aug",
    "September": "sep", "October": "oct", "November": "nov", "December": "dec"
}
day_options = {"Monday": "mon", "Tuesday": "tue", "Wednesday": "wed", "Thursday": "thu", "Friday": "fri"}
poutcome_options = {"Nonexistent": "nonexistent", "Failure": "failure", "Success": "success"}

# User inputs
job_value = job_options[st.selectbox("Job", list(job_options.keys()))]
marital_value = marital_options[st.selectbox("Marital Status", list(marital_options.keys()))]
education_value = education_options[st.selectbox("Education", list(education_options.keys()))]
default_value = default_options[st.selectbox("Credit in Default?", list(default_options.keys()))]
housing_value = housing_options[st.selectbox("Has Housing Loan?", list(housing_options.keys()))]
loan_value = loan_options[st.selectbox("Has Personal Loan?", list(loan_options.keys()))]
contact_value = contact_options[st.selectbox("Contact Communication Type", list(contact_options.keys()))]
month_value = month_options[st.selectbox("Last Contact Month", list(month_options.keys()))]
day_value = day_options[st.selectbox("Day of Week", list(day_options.keys()))]
poutcome_value = poutcome_options[st.selectbox("Outcome of Previous Campaign", list(poutcome_options.keys()))]

# Numeric inputs
age = st.number_input("Age", min_value=18, max_value=100, value=35)
duration = st.number_input("Duration of Last Contact (seconds)", min_value=0)
campaign = st.number_input("Number of Contacts During Campaign", min_value=1)
pdays = st.number_input("Days Since Last Contact (previous campaign)", min_value=-1)
previous = st.number_input("Number of Previous Contacts", min_value=0)
emp_var_rate = st.number_input("Employment Variation Rate", format="%.2f")
cons_price_idx = st.number_input("Consumer Price Index", format="%.2f")
cons_conf_idx = st.number_input("Consumer Confidence Index", format="%.2f")
euribor3m = st.number_input("Euribor 3 Month Rate", format="%.3f")
nr_employed = st.number_input("Number of Employees", format="%.1f")

# Construct input DataFrame
input_dict = {
    'age': age,
    'job': job_value,
    'marital': marital_value,
    'education': education_value,
    'default': default_value,
    'housing': housing_value,
    'loan': loan_value,
    'contact': contact_value,
    'month': month_value,
    'day_of_week': day_value,
    'duration': duration,
    'campaign': campaign,
    'pdays': pdays,
    'previous': previous,
    'poutcome': poutcome_value,
    'emp.var.rate': emp_var_rate,
    'cons.price.idx': cons_price_idx,
    'cons.conf.idx': cons_conf_idx,
    'euribor3m': euribor3m,
    'nr.employed': nr_employed
}
input_df = pd.DataFrame([input_dict])

# Predict button
if st.button("Predict Term Deposit Subscription"):
    prediction = pipeline.predict(input_df)[0]
    prob = pipeline.predict_proba(input_df)[0][1]

    if prediction == 1:
        st.success(f"✅ Customer is likely to SUBSCRIBE (probability: {prob:.2f})")
    else:
        st.warning(f"❌ Customer is NOT likely to subscribe (probability: {prob:.2f})")

