import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Load saved model & references
model = joblib.load("naive_bayes_model.joblib")
col_info = joblib.load("unique_elements_dict_naivebayes.joblib")

model_columns = joblib.load("model_columns.joblib")

# -----------------------
# UI: User Input Form
# -----------------------
with st.form("predict_form"):
    st.title("📱🏦 Financial Inclusion in Africa App detector")

    # Dropdown selectors for categorical values
    country = st.selectbox("Country", col_info["country"])
    year = st.selectbox("Year", col_info["year"])
    
    location_type = st.selectbox("Location Type", col_info["location_type"])
    cellphone_access = st.selectbox("Cellphone Access", col_info["cellphone_access"])
    
    # Sliders for numerical values
    household_size = st.slider("Household Size", min_value=min(col_info["household_size"]), max_value=max(col_info["household_size"]), value=min(col_info["household_size"]))
    age_of_respondent = st.slider("Age of Respondent", min_value=min(col_info["age_of_respondent"]), max_value=max(col_info["age_of_respondent"]), value=min(col_info["age_of_respondent"]))
    
    # Dropdown selectors for relationship and marital status
    gender_of_respondent = st.selectbox("Gender of Respondent", col_info["gender_of_respondent"])
    relationship_with_head = st.selectbox("Relationship with Head of Household", col_info["relationship_with_head"])
    marital_status = st.selectbox("Marital Status", col_info["marital_status"])
    
    # Dropdown for education level and job type
    education_level = st.selectbox("Education Level", col_info["education_level"])
    job_type = st.selectbox("Job Type", col_info["job_type"])

    submitted = st.form_submit_button("Predict")

# -----------------------
# Data Transformation & Prediction
# -----------------------
if submitted:
    # 1. Raw input to DataFrame
    df = pd.DataFrame([{
        "year": year,
        "cellphone_access": cellphone_access,
        "household_size": household_size,
        "age_of_respondent": age_of_respondent,
        "country": country,
        "location_type": location_type,
        "gender_of_respondent": gender_of_respondent,
        "relationship_with_head": relationship_with_head,
        "marital_status": marital_status,
        "education_level": education_level,
        "job_type": job_type
    }])

    # 2. Binary mapping
    binary_map = {"Yes": 1, "No": 0}
    df["cellphone_access"] = df["cellphone_access"].map(binary_map)

    # 3. One-hot encoding (drop bank_account here)
    categorical_cols = [
        'country',
        'location_type',
        'gender_of_respondent',
        'relationship_with_head',
        'marital_status',
        'education_level',
        'job_type'
    ]

    df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    df_encoded = df_encoded.astype(int)

    # 4. Reindex to match model's expected input columns
    df_encoded = df_encoded.reindex(columns=model_columns, fill_value=0)

    # 5. Predict
    prediction = model.predict(df_encoded)[0]
    prob = model.predict_proba(df_encoded)[0][1]

    st.success("✅ Has a Bank Account" if prediction == 1 else "❌ No Bank Account")
    st.info(f"📈 Probability: {prob:.2%}")
