import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# Page configuration
st.set_page_config(
    page_title="Financial Inclusion Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        color: #1E3A8A;
        margin-top: 1rem;
        margin-bottom: 1rem;
        padding-bottom: 0.5rem;
        border-bottom: 1px solid #E5E7EB;
    }
    .prediction-box-positive {
        background-color: #DCFCE7;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #22C55E;
        margin: 1rem 0;
    }
    .prediction-box-negative {
        background-color: #FEE2E2;
        padding: 1.5rem;
        border-radius: 0.5rem;
        border-left: 5px solid #EF4444;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #EFF6FF;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    .stProgress > div > div > div > div {
        background-color: #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# Load saved model & references
@st.cache_resource
def load_model_data():
    model = joblib.load("naive_bayes_model.joblib")
    col_info = joblib.load("unique_elements_dict_naivebayes.joblib")
    model_columns = joblib.load("model_columns.joblib")
    return model, col_info, model_columns

model, col_info, model_columns = load_model_data()

# Sidebar with information about financial inclusion
with st.sidebar:
    st.image("https://www.worldbank.org/content/dam/photos/780x439/2022/apr/Financial-Inclusion-Africa-780.jpg", use_column_width=True)
    st.markdown("## About Financial Inclusion")
    st.markdown("""
    Financial inclusion means that individuals and businesses have access to useful and affordable financial products and services that meet their needs – transactions, payments, savings, credit and insurance – delivered in a responsible and sustainable way.
    
    This app predicts whether an individual in East Africa is likely to have a bank account based on demographic information.
    """)
    
    st.markdown("### Dataset Information")
    st.markdown("""
    - **Source**: Zindi platform
    - **Coverage**: ~33,600 individuals across East Africa
    - **Features**: Demographics and financial service usage
    - **Target**: Bank account ownership
    """)
    
    st.markdown("### How to Use")
    st.markdown("""
    1. Fill in the demographic information in the form
    2. Click 'Predict' to see the likelihood of bank account ownership
    3. Review the prediction and probability score
    """)

# Main content
st.markdown('<h1 class="main-header">Financial Inclusion in Africa Predictor</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict bank account ownership based on demographic factors</p>', unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2 = st.tabs(["Make Prediction", "Model Information"])

with tab1:
    # Form in a card-like container
    with st.container():
        st.markdown('<h2 class="section-header">Enter Individual Information</h2>', unsafe_allow_html=True)
        
        # Create columns for better layout
        col1, col2 = st.columns(2)
        
        with st.form("predict_form"):
            # Location Information
            with col1:
                st.markdown("#### Location Information")
                country = st.selectbox("Country", col_info["country"])
                location_type = st.selectbox("Location Type", col_info["location_type"])
                year = st.selectbox("Year", col_info["year"])
            
            # Personal Information
            with col2:
                st.markdown("#### Personal Information")
                gender_of_respondent = st.selectbox("Gender", col_info["gender_of_respondent"])
                age_of_respondent = st.slider("Age", min_value=min(col_info["age_of_respondent"]), 
                                             max_value=max(col_info["age_of_respondent"]), 
                                             value=min(col_info["age_of_respondent"]))
                marital_status = st.selectbox("Marital Status", col_info["marital_status"])
            
            # Household Information
            col3, col4 = st.columns(2)
            with col3:
                st.markdown("#### Household Information")
                household_size = st.slider("Household Size", min_value=min(col_info["household_size"]), 
                                          max_value=max(col_info["household_size"]), 
                                          value=min(col_info["household_size"]))
                relationship_with_head = st.selectbox("Relationship with Head", col_info["relationship_with_head"])
            
            # Socioeconomic Information
            with col4:
                st.markdown("#### Socioeconomic Information")
                education_level = st.selectbox("Education Level", col_info["education_level"])
                job_type = st.selectbox("Job Type", col_info["job_type"])
                cellphone_access = st.selectbox("Cellphone Access", col_info["cellphone_access"])
            
            # Submit button with better styling
            submitted = st.form_submit_button("Predict Bank Account Ownership", use_container_width=True)
    
    # Prediction section
    if submitted:
        # Show a spinner during prediction
        with st.spinner("Analyzing data..."):
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
    
            # 3. One-hot encoding
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
        
        # Display prediction results
        st.markdown('<h2 class="section-header">Prediction Results</h2>', unsafe_allow_html=True)
        
        # Create two columns for results and visualization
        result_col, viz_col = st.columns([3, 2])
        
        with result_col:
            if prediction == 1:
                st.markdown(f"""
                <div class="prediction-box-positive">
                    <h3>✅ Likely to Have a Bank Account</h3>
                    <p>This individual is predicted to have access to banking services.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="prediction-box-negative">
                    <h3>❌ Unlikely to Have a Bank Account</h3>
                    <p>This individual is predicted to lack access to banking services.</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Probability gauge
            st.markdown("### Confidence Level")
            st.progress(prob)
            st.markdown(f"**Probability: {prob:.2%}**")
            
            # Key factors
            st.markdown("### Key Factors")
            st.markdown("""
            <div class="info-box">
                <p>The prediction is based on several key factors:</p>
                <ul>
                    <li>Location and country of residence</li>
                    <li>Education level and job type</li>
                    <li>Age and household characteristics</li>
                    <li>Access to technology (cellphone)</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)
        
        with viz_col:
            # Create a simple visualization of the input factors
            st.markdown("### Input Profile")
            
            # Create a radar chart of key numeric and binary factors
            fig, ax = plt.subplots(figsize=(4, 4))
            
            # Normalize age and household size for the chart
            max_age = max(col_info["age_of_respondent"])
            max_household = max(col_info["household_size"])
            
            # Create data for radar chart (normalized values)
            categories = ['Age', 'Household Size', 'Cellphone', 'Urban', 'Education']
            values = [
                age_of_respondent / max_age,
                household_size / max_household,
                1 if cellphone_access == "Yes" else 0,
                1 if location_type == "Urban" else 0,
                0.2 if education_level == "No formal education" else 
                0.4 if education_level == "Primary education" else
                0.6 if education_level == "Secondary education" else
                0.8 if education_level == "Tertiary education" else 1.0
            ]
            
            # Close the loop for the radar chart
            values.append(values[0])
            categories.append(categories[0])
            
            # Plot the radar chart
            ax.plot(categories, values, marker='o')
            ax.fill(categories, values, alpha=0.25)
            ax.set_ylim(0, 1)
            plt.tight_layout()
            
            st.pyplot(fig)

with tab2:
    st.markdown('<h2 class="section-header">About the Model</h2>', unsafe_allow_html=True)
    
    # Model information
    st.markdown("""
    ### Model Type: Naive Bayes
    
    This application uses a Naive Bayes classifier to predict bank account ownership. The model was trained on demographic data from approximately 33,600 individuals across East Africa.
    
    ### Key Features
    
    The model considers several important factors when making predictions:
    
    - **Geographic factors**: Country and urban/rural location
    - **Demographic factors**: Age, gender, marital status
    - **Socioeconomic factors**: Education level, job type
    - **Household characteristics**: Household size, relationship with head
    - **Technology access**: Cellphone ownership
    
    ### Model Performance
    
    The model has been evaluated on a test dataset and demonstrates good predictive performance for financial inclusion.
    """)
    
    # Sample data visualization
    st.markdown('<h2 class="section-header">Financial Inclusion Insights</h2>', unsafe_allow_html=True)
    
    # Create some example visualizations
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### Financial Inclusion by Country")
        # Example data - replace with actual statistics if available
        countries = ["Kenya", "Rwanda", "Tanzania", "Uganda"]
        inclusion_rates = [0.42, 0.38, 0.23, 0.28]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.bar(countries, inclusion_rates, color="#1E3A8A")
        ax.set_ylabel("Bank Account Ownership Rate")
        ax.set_ylim(0, 0.5)
        for i, v in enumerate(inclusion_rates):
            ax.text(i, v + 0.02, f"{v:.0%}", ha='center')
        plt.tight_layout()
        st.pyplot(fig)
    
    with col2:
        st.markdown("### Factors Affecting Financial Inclusion")
        # Example data - replace with actual statistics if available
        factors = ["Urban Location", "Higher Education", "Cellphone Access", "Formal Employment"]
        impact = [0.65, 0.78, 0.52, 0.71]
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(factors, impact, color="#4F46E5")
        ax.set_xlabel("Correlation with Bank Account Ownership")
        ax.set_xlim(0, 1)
        for i, v in enumerate(impact):
            ax.text(v + 0.02, i, f"{v:.0%}", va='center')
        plt.tight_layout()
        st.pyplot(fig)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #6B7280; font-size: 0.8rem;">
    <p>Financial Inclusion in Africa Predictor | Developed for educational purposes</p>
    <p>Data source: Zindi platform | Model: Naive Bayes Classifier</p>
</div>
""", unsafe_allow_html=True)
