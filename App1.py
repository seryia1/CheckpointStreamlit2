import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import time
from datetime import datetime

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
    
    /* Dashboard card styling */
    .dashboard-card {
        background-color: white;
        border-radius: 0.5rem;
        padding: 1rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        margin-bottom: 1rem;
        border-top: 4px solid #1E3A8A;
    }
    
    /* Video responsive container */
    .video-container {
        position: relative;
        padding-bottom: 56.25%;
        height: 0;
        overflow: hidden;
        max-width: 100%;
    }
    .video-container iframe {
        position: absolute;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
    }
    
    /* Gallery styling */
    .gallery {
        display: flex;
        flex-wrap: wrap;
        gap: 10px;
    }
    .gallery-item {
        flex: 1 0 200px;
        border-radius: 5px;
        overflow: hidden;
    }
    .gallery-item img {
        width: 100%;
        height: auto;
        transition: transform 0.3s ease;
    }
    .gallery-item img:hover {
        transform: scale(1.05);
    }
    
    /* Infographic styling */
    .infographic {
        background-color: #F8FAFC;
        border-radius: 0.5rem;
        padding: 1.5rem;
        margin: 1rem 0;
    }
    
    /* Custom tab styling */
    .custom-tab {
        background-color: #F1F5F9;
        padding: 10px 15px;
        border-radius: 5px 5px 0 0;
        margin-right: 5px;
        cursor: pointer;
        border: 1px solid #E2E8F0;
        border-bottom: none;
    }
    .custom-tab.active {
        background-color: white;
        font-weight: bold;
        border-top: 3px solid #1E3A8A;
    }
    .custom-tab-content {
        padding: 20px;
        border: 1px solid #E2E8F0;
        border-radius: 0 5px 5px 5px;
        background-color: white;
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

# ----- 11. RICH MEDIA INTEGRATION -----
# Sample videos about financial inclusion
financial_inclusion_videos = [
    {
        "title": "What is Financial Inclusion?",
        "url": "https://www.youtube.com/watch?v=cAxFBzHV6tc",
        "description": "An overview of financial inclusion and its importance in developing economies."
    },
    {
        "title": "Digital Financial Inclusion in Africa",
        "url": "https://www.youtube.com/watch?v=4rtJHeR2EIc",
        "description": "How digital technologies are transforming financial access across Africa."
    },
    {
        "title": "Mobile Money Revolution in Africa",
        "url": "https://www.youtube.com/watch?v=Ava9I6S2leg",
        "description": "The impact of mobile money services on financial inclusion in Africa."
    }
]

# Sample infographics about financial inclusion - UPDATED WITH NEW URLS
financial_inclusion_infographics = [
    {
        "title": "Financial Inclusion Factors",
        "image": "https://fastercapital.co/i/Financial-Inclusion-Score--How-to-Access-and-Benefit-from-Financial-Services-and-Products--Factors-Affecting-Financial-Inclusion-Score.webp",
        "description": "Key factors affecting financial inclusion globally."
    },
    {
        "title": "Mobile Money in Africa",
        "image": "https://cdn.statcdn.com/Infographic/images/normal/25713.jpeg",
        "description": "The growth of mobile money accounts in Sub-Saharan Africa."
    },
    {
        "title": "Gender Gap in Financial Inclusion",
        "image": "https://blogs.worldbank.org/content/dam/sites/blogs/img/detail/mgr/unbanked.jpg",
        "description": "Gender disparities in access to financial services."
    }
]



# ----- 12. ADVANCED DATA DISPLAYS -----
# Sample data for visualizations
@st.cache_data
def load_sample_data():
    # Sample time series data for financial inclusion trends
    years = list(range(2010, 2023))
    kenya_trend = [0.25, 0.28, 0.32, 0.35, 0.38, 0.42, 0.45, 0.48, 0.52, 0.55, 0.58, 0.62, 0.65]
    rwanda_trend = [0.18, 0.20, 0.23, 0.25, 0.28, 0.32, 0.35, 0.38, 0.42, 0.45, 0.48, 0.52, 0.55]
    tanzania_trend = [0.12, 0.14, 0.16, 0.18, 0.20, 0.23, 0.25, 0.28, 0.30, 0.33, 0.35, 0.38, 0.40]
    uganda_trend = [0.15, 0.17, 0.19, 0.22, 0.24, 0.27, 0.30, 0.33, 0.36, 0.39, 0.42, 0.45, 0.48]
    
    time_series_df = pd.DataFrame({
        'Year': years,
        'Kenya': kenya_trend,
        'Rwanda': rwanda_trend,
        'Tanzania': tanzania_trend,
        'Uganda': uganda_trend
    })
    
    # Sample data for heatmap of financial inclusion factors
    heatmap_data = pd.DataFrame({
        'Country': ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'] * 5,
        'Factor': ['Mobile Money', 'Mobile Money', 'Mobile Money', 'Mobile Money',
                  'Bank Account', 'Bank Account', 'Bank Account', 'Bank Account',
                  'Digital Payments', 'Digital Payments', 'Digital Payments', 'Digital Payments',
                  'Savings', 'Savings', 'Savings', 'Savings',
                  'Credit', 'Credit', 'Credit', 'Credit'],
        'Adoption Rate': [0.72, 0.65, 0.58, 0.62, 
                         0.55, 0.48, 0.35, 0.42,
                         0.68, 0.58, 0.45, 0.52,
                         0.45, 0.38, 0.30, 0.35,
                         0.32, 0.25, 0.18, 0.22]
    })
    
    # Sample data for demographic comparison
    demographic_data = pd.DataFrame({
        'Demographic': ['Urban', 'Rural', 'Male', 'Female', 'Primary Education', 'Secondary Education', 'Tertiary Education'],
        'Bank Account': [0.65, 0.32, 0.58, 0.45, 0.35, 0.58, 0.82],
        'Mobile Money': [0.78, 0.52, 0.72, 0.68, 0.58, 0.75, 0.88]
    })
    
    # Sample data for geographic heatmap
    geo_data = pd.DataFrame({
        'Country': ['Kenya', 'Rwanda', 'Tanzania', 'Uganda'],
        'Latitude': [1.2921, -1.9403, -6.3690, 1.3733],
        'Longitude': [36.8219, 29.8739, 34.8888, 32.2903],
        'Bank_Account_Rate': [0.55, 0.48, 0.35, 0.42],
        'Mobile_Money_Rate': [0.72, 0.65, 0.58, 0.62]
    })
    
    return time_series_df, heatmap_data, demographic_data, geo_data

time_series_df, heatmap_data, demographic_data, geo_data = load_sample_data()

# Create advanced visualizations
@st.cache_data
def create_time_series_chart():
    fig = px.line(time_series_df, x='Year', y=['Kenya', 'Rwanda', 'Tanzania', 'Uganda'],
                 title='Financial Inclusion Trends in East Africa (2010-2022)',
                 labels={'value': 'Account Ownership Rate', 'variable': 'Country'},
                 color_discrete_sequence=['#1E3A8A', '#4F46E5', '#7C3AED', '#EC4899'])
    fig.update_layout(
        legend_title_text='Country',
        xaxis_title='Year',
        yaxis_title='Account Ownership Rate',
        hovermode='x unified',
        height=500
    )
    return fig

@st.cache_data
def create_heatmap():
    pivot_data = heatmap_data.pivot(index='Factor', columns='Country', values='Adoption Rate')
    fig = px.imshow(pivot_data, 
                   labels=dict(x="Country", y="Financial Service", color="Adoption Rate"),
                   x=pivot_data.columns, 
                   y=pivot_data.index,
                   color_continuous_scale='Blues',
                   title='Financial Services Adoption Heatmap')
    fig.update_layout(height=500)
    return fig

@st.cache_data
def create_demographic_comparison():
    fig = px.bar(demographic_data, x='Demographic', y=['Bank Account', 'Mobile Money'],
                barmode='group', title='Financial Services by Demographic Group',
                color_discrete_sequence=['#1E3A8A', '#4F46E5'])
    fig.update_layout(
        xaxis_title='Demographic Group',
        yaxis_title='Adoption Rate',
        legend_title='Service Type',
        height=500
    )
    return fig

@st.cache_data
def create_geo_visualization():
    fig = px.scatter_mapbox(geo_data, 
                           lat='Latitude', 
                           lon='Longitude', 
                           size='Bank_Account_Rate',
                           color='Mobile_Money_Rate',
                           hover_name='Country', 
                           color_continuous_scale='Viridis',
                           size_max=25, 
                           zoom=3,
                           mapbox_style='carto-positron',
                           title='Financial Inclusion Across East Africa')
    fig.update_layout(height=600)
    return fig

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
    
    # Dashboard customization options
    st.markdown("### Dashboard Settings")
    show_prediction_form = st.checkbox("Show Prediction Form", value=True)
    show_visualizations = st.checkbox("Show Data Visualizations", value=True)
    show_media = st.checkbox("Show Media Resources", value=True)
    
    # Theme selection
    st.markdown("### Theme")
    theme = st.selectbox("Select Theme", ["Default Blue", "Green", "Purple", "Orange"])
    
    # Apply theme colors
    if theme == "Green":
        primary_color = "#047857"
        secondary_color = "#10B981"
    elif theme == "Purple":
        primary_color = "#6D28D9"
        secondary_color = "#8B5CF6"
    elif theme == "Orange":
        primary_color = "#D97706"
        secondary_color = "#F59E0B"
    else:  # Default Blue
        primary_color = "#1E3A8A"
        secondary_color = "#4F46E5"
    
    st.markdown(f"""
    <style>
        .section-header {{
            color: {primary_color};
        }}
        .stProgress > div > div > div > div {{
            background-color: {primary_color};
        }}
        .dashboard-card {{
            border-top: 4px solid {primary_color};
        }}
        .custom-tab.active {{
            border-top: 3px solid {primary_color};
        }}
    </style>
    """, unsafe_allow_html=True)

# ----- 21. DASHBOARD LAYOUT -----
# Main content
st.markdown('<h1 class="main-header">Financial Inclusion in Africa Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Predict bank account ownership and explore financial inclusion insights</p>', unsafe_allow_html=True)

# Create dashboard tabs
dashboard_tabs = ["Prediction", "Data Insights", "Media Resources", "About"]
selected_tab = st.radio("", dashboard_tabs, horizontal=True, label_visibility="collapsed")

# Prediction Tab
if selected_tab == "Prediction" and show_prediction_form:
    # Form in a card-like container
    with st.container():
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)
    
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
            
            # Add a slight delay for animation effect
            time.sleep(0.5)
        
        # Display prediction results in a dashboard card
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
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
        st.markdown('</div>', unsafe_allow_html=True)

# Data Insights Tab
elif selected_tab == "Data Insights" and show_visualizations:
    # Create a dashboard of visualizations
    st.markdown('<h2 class="section-header">Financial Inclusion Data Insights</h2>', unsafe_allow_html=True)
    
    # Create visualization tabs
    viz_tabs = ["Trends", "Comparisons", "Geographic", "Factors"]
    viz_tab = st.radio("Select visualization type:", viz_tabs, horizontal=True)
    
    if viz_tab == "Trends":
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Financial Inclusion Trends Over Time")
        st.plotly_chart(create_time_series_chart(), use_container_width=True)
        st.markdown("""
        This chart shows the growth of financial inclusion (measured by account ownership) 
        across East African countries from 2010 to 2022. Kenya has consistently led the region 
        in financial inclusion, with significant growth across all countries over the past decade.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif viz_tab == "Comparisons":
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Demographic Comparison of Financial Services")
        st.plotly_chart(create_demographic_comparison(), use_container_width=True)
        st.markdown("""
        This chart compares bank account ownership and mobile money usage across different 
        demographic groups. Note that mobile money adoption is generally higher than traditional 
        banking across all demographics, with the gap being particularly pronounced in rural areas.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif viz_tab == "Geographic":
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Geographic Distribution of Financial Inclusion")
        st.plotly_chart(create_geo_visualization(), use_container_width=True)
        st.markdown("""
        This map shows the geographic distribution of financial inclusion across East Africa.
        The size of each bubble represents bank account ownership rates, while the color 
        represents mobile money adoption rates. Kenya leads in both metrics.
        """)
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif viz_tab == "Factors":
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Financial Services Adoption Heatmap")
        st.plotly_chart(create_heatmap(), use_container_width=True)
        st.markdown("""
        This heatmap shows the adoption rates of different financial services across 
        East African countries. Mobile money has the highest adoption rates across all countries,
        while access to credit remains relatively low.
        """)
        st.markdown('</div>', unsafe_allow_html=True)

# Media Resources Tab
elif selected_tab == "Media Resources" and show_media:
    st.markdown('<h2 class="section-header">Financial Inclusion Media Resources</h2>', unsafe_allow_html=True)
    
    # Create media tabs
    media_tabs = ["Videos", "Infographics", "Image Gallery"]
    media_tab = st.radio("Select media type:", media_tabs, horizontal=True)
    
    if media_tab == "Videos":
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Educational Videos on Financial Inclusion")
        
        for i, video in enumerate(financial_inclusion_videos):
            st.markdown(f"#### {video['title']}")
            st.markdown(f"""
            <div class="video-container">
                <iframe src="{video['url']}" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>
            </div>
            """, unsafe_allow_html=True)
            st.markdown(f"{video['description']}")
            if i < len(financial_inclusion_videos) - 1:
                st.markdown("---")
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif media_tab == "Infographics":
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Financial Inclusion Infographics")
        
        for i, infographic in enumerate(financial_inclusion_infographics):
            st.markdown(f"#### {infographic['title']}")
            st.image(infographic['image'], use_column_width=True)
            st.markdown(f"{infographic['description']}")
            if i < len(financial_inclusion_infographics) - 1:
                st.markdown("---")
        st.markdown('</div>', unsafe_allow_html=True)
    
    elif media_tab == "Image Gallery":
        st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
        st.markdown("### Financial Inclusion Image Gallery")
        
        st.markdown("""
        <div class="gallery">
        """, unsafe_allow_html=True)
        
        # Display gallery in rows of 3
        cols = st.columns(3)
        for i, image_url in enumerate(financial_inclusion_gallery):
            with cols[i % 3]:
                st.image(image_url, use_column_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

# About Tab
elif selected_tab == "About":
    st.markdown('<div class="dashboard-card">', unsafe_allow_html=True)
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
    - **Househol  gender, marital status
    - **Socioeconomic factors**: Education level, job type
    - **Household characteristics**: Household size, relationship with head
    - **Technology access**: Cellphone ownership
    
    ### Model Performance
    
    The model has been evaluated on a test dataset and demonstrates good predictive performance for financial inclusion.
    """)
    
    st.markdown('<h2 class="section-header">About Financial Inclusion</h2>', unsafe_allow_html=True)
    st.markdown("""
    Financial inclusion is the delivery of financial services at affordable costs to disadvantaged and low-income segments of society. It is a key enabler to reducing poverty and boosting prosperity in developing economies.
    
    ### Why Financial Inclusion Matters
    
    - **Poverty Reduction**: Access to financial services helps people escape poverty by facilitating investments in education, health, and business
    - **Economic Growth**: Broadens the base of savers and promotes capital accumulation
    - **Gender Equality**: Helps women assert economic power and make financial decisions
    - **Food Security**: Enables farmers to invest in higher-yielding crops and better equipment
    
    ### Financial Inclusion in Africa
    
    Africa has made significant strides in financial inclusion, largely driven by mobile money services. However, challenges remain:
    
    - **Infrastructure Gaps**: Limited physical banking infrastructure in rural areas
    - **Documentation Requirements**: Many lack the formal documentation needed for traditional banking
    - **Financial Literacy**: Limited understanding of financial products and services
    - **Gender Gap**: Women have less access to financial services than men
    
    This application aims to help identify individuals who may be excluded from the financial system, enabling targeted interventions to improve access.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# Dashboard footer
st.markdown("---")
st.markdown(f"""
<div style="text-align: center; color: #6B7280; font-size: 0.8rem;">
    <p>Financial Inclusion in Africa Dashboard | Last updated: {datetime.now().strftime('%B %d, %Y')}</p>
    <p>Data source: Zindi platform | Model: Naive Bayes Classifier</p>
</div>
""", unsafe_allow_html=True)
