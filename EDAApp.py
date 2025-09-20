# ------------------------
# Streamlit Vaccine Dashboard
# ------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from sqlalchemy import create_engine
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")

# ------------------------
# Database connection
# ------------------------
server = r'UZZII\SQLEXPRESS'
database = "PHARMA"

engine = create_engine(
    f"mssql+pyodbc://@{server}/{database}?trusted_connection=yes&driver=ODBC+Driver+17+for+SQL+Server"
)

# ------------------------
# User credentials
# ------------------------
USER_CREDENTIALS = {
    "admin": "admin123",
    "user": "user123"
}

# Initialize session state
if "authenticated" not in st.session_state:
    st.session_state["authenticated"] = False

# ------------------------
# Login function
# ------------------------
def login():
    st.header("🔐 Welcome to Vaccine Uptake Hub")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    login_button = st.button("Login")
    
    if login_button:
        if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
            st.session_state["authenticated"] = True
            st.success(f"Welcome, {username}!")
            st.rerun()
        else:
            st.error("Invalid username or password")

# ------------------------
# Main app function
# ------------------------
def main_app():
    # Load data
    df = pd.read_sql('SELECT * FROM VACCINES', engine)      

    st.sidebar.title("🧭 Navigation")
    selection = st.sidebar.radio("Go to Section",["Home Page","Key Metrics","Understand Vaccine Coverage","Demographic Insights","Additional Visualizations"])
    
                                          
    if selection == "Home Page":
        st.title("💉 Vaccine Analysis & Prediction Dashboard")
        st.markdown(""" 
        ## Welcome to the Vaccine Uptake Hub!
### 🎯 Objectives
This dashboard is designed to help you:
- Monitor key performance metrics
- Understand Vaccine Coverage
- Demographic Insights
- Additional Visualizations
### 🚀 Let's Get Started                    
         
       """) 
        

        st.markdown('### 🗂️ Analysis Overview')    
    
        # Data as a list of dictionaries               
        data = [
        {"Section": "Key Metrics", "Description": "Understand overall Vaccine Coverage with essential performance indicators."},
        {"Section": "Vaccine Coverage", "Description": "Analyze vaccine uptake across different demographics and regions."},
        {"Section": "Demographic Insights", "Description": "Explore vaccination trends based on age"},
        {"Section": "Additional Visualizations", "Description": "Dive deeper with correlation heatmaps and concern/knowledge analysis."},
        ]
 
        # Create DataFrame
        df_sections = pd.DataFrame(data) 

        # Removes the index and keeps column headers
        st.table(df_sections)


    if selection == "Key Metrics":
        st.header("📊 Vaccine Intake at a Glance")

        st.subheader("📌 Key Metrics")

        # ✅ KPI Calculations (using booleans instead of Yes/No)
        h1n1_vacc_rate = df['h1n1_vaccine'].mean() * 100
        seasonal_vacc_rate = df['seasonal_vaccine'].mean() * 100
        Both_Vaccinated_Percentage = ((df['h1n1_vaccine']) & (df['seasonal_vaccine'])).mean() * 100
        Neither_Vaccinated_Percentage = ((~df['h1n1_vaccine']) & (~df['seasonal_vaccine'])).mean() * 100
        Count_of_H1N1_Vaccinated = df['h1n1_vaccine'].sum()
        Count_of_Seasonal_Vaccinated = df['seasonal_vaccine'].sum()
        Count_of_H1N1_Knowledgeable = (df['h1n1_knowledge'] == 2).sum()
        Count_of_Seasonal_Knowledgeable = (df['opinion_seas_risk'] > 2).sum()
        total_respondents = df['respondent_id'].nunique()

        # These columns are booleans too
        doctor_recc_rate = df['doctor_recc_h1n1'].mean() * 100 if 'doctor_recc_h1n1' in df.columns else None
        health_insurance_rate = df['health_insurance'].mean() * 100 if 'health_insurance' in df.columns else None
        chronic_condition_rate = df['chronic_med_condition'].mean() * 100 if 'chronic_med_condition' in df.columns else None
        health_worker_rate = df['health_worker'].mean() * 100 if 'health_worker' in df.columns else None

        # Display metrics in rows of 3
        col6, col7, col8 = st.columns(3)
        col6.metric("H1N1 Vaccine %", f"{h1n1_vacc_rate:.2f}%")
        col7.metric("Seasonal Vaccine %", f"{seasonal_vacc_rate:.2f}%")
        col8.metric("Both Vaccinated %", f"{Both_Vaccinated_Percentage:.2f}%")

        col9, col10, col11 = st.columns(3)
        col9.metric("No Vaccine %", f"{Neither_Vaccinated_Percentage:.2f}%")
        col10.metric("Count of H1N1 Vaccinated", f"{Count_of_H1N1_Vaccinated}")
        col11.metric("Count of Seasonal Vaccinated", f"{Count_of_Seasonal_Vaccinated}")

        col12, col13, col14 = st.columns(3)
        col12.metric("Count of H1N1 Knowledgeable", f"{Count_of_H1N1_Knowledgeable}")
        col13.metric("Count of Seasonal Knowledgeable", f"{Count_of_Seasonal_Knowledgeable}")
        col14.metric("Total Respondents", f"{total_respondents}")

        col15, col16, col17 = st.columns(3)
        if doctor_recc_rate is not None:
            col15.metric("Doctor Recommendation Rate", f"{doctor_recc_rate:.2f}%")
        if health_insurance_rate is not None:
            col16.metric("Health Insurance Rate", f"{health_insurance_rate:.2f}%")
        if health_worker_rate is not None:
            col17.metric("Health Worker Rate", f"{health_worker_rate:.2f}%")

        st.markdown("---")

    if selection == "Understand Vaccine Coverage":

        st.header("🌍 Vaccine Coverage Analysis")
        # Define region lookup dictionary
        region_lookup = {
            "lzgpxyit": "Boston (New England)",
            "fpwskwrf": "New York",
            "mlyzmhmf": "Philadelphia",
            "bhuqouqj": "Atlanta",
            "oxchjgsf": "Chicago",
            "atmpeygn": "Dallas",
            "dqpwygqj": "Kansas City",
            "kbazzjca": "Denver",
            "qufhixun": "San Francisco",
            "lrircsnp": "Seattle"
        }

        # Map coded region values to readable names
        df["region_name"] = df["hhs_geo_region"].map(region_lookup)

        # Sidebar Filters
        employment = df['employment_status'].dropna().sort_values().unique()
        region = df['region_name'].dropna().sort_values().unique()

        st.sidebar.title("🔍 Filters")
        selected_employment = st.sidebar.multiselect("Employment Status", employment)
        selected_region = st.sidebar.multiselect("Region", region)

        # Apply filters to DataFrame
        final_df = df.copy()
        if selected_employment:
            final_df = final_df[final_df["employment_status"].isin(selected_employment)]
        if selected_region:
            final_df = final_df[final_df["region_name"].isin(selected_region)]

        # =======================
        # 📈 Vaccine Uptake by Age Group
        # =======================
        st.subheader("📈 Vaccine Uptake by Age Group")
        age_vacc = final_df.groupby('age_group')[['h1n1_vaccine','seasonal_vaccine']].mean().reset_index()
        age_vacc[['h1n1_vaccine','seasonal_vaccine']] *= 100  

        fig1 = px.bar(
            age_vacc,
            x='age_group',
            y=['h1n1_vaccine','seasonal_vaccine'],
            barmode='group',
            labels={'value': 'Uptake Rate (%)', 'variable': 'Vaccine Type'},
            text_auto='.2f'
        )
        fig1.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
        fig1.update_layout(yaxis_title="Uptake Rate (%)")
        st.plotly_chart(fig1)

        # =======================
        # 📈 Vaccine Uptake by Income Level
        # =======================
        st.subheader("📈 Vaccine Uptake by Income Level")
        income_vacc = final_df.groupby('income_poverty')[['h1n1_vaccine','seasonal_vaccine']].mean().reset_index()
        income_vacc[['h1n1_vaccine','seasonal_vaccine']] *= 100

        fig2 = px.bar(
            income_vacc,
            x='income_poverty',
            y=['h1n1_vaccine','seasonal_vaccine'],
            barmode='group',
            labels={'value': 'Uptake Rate (%)', 'variable': 'Vaccine Type'},
            text_auto='.2f'
        )
        fig2.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
        fig2.update_layout(yaxis_title="Uptake Rate (%)")
        st.plotly_chart(fig2)

        # =======================
        # 📈 Vaccine Uptake by Education Level
        # =======================
        st.subheader("📈 Vaccine Uptake by Education Level")
        edu_vacc = final_df.groupby('education')[['h1n1_vaccine','seasonal_vaccine']].mean().reset_index()
        edu_vacc[['h1n1_vaccine','seasonal_vaccine']] *= 100

        fig3 = px.bar(
            edu_vacc,
            x='education',
            y=['h1n1_vaccine','seasonal_vaccine'],
            barmode='group',
            labels={'value': 'Uptake Rate (%)', 'variable': 'Vaccine Type'},
            text_auto='.2f'
        )
        fig3.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
        fig3.update_layout(yaxis_title="Uptake Rate (%)")
        st.plotly_chart(fig3)

        st.markdown("---")

        # =======================
        # 📈 Vaccine Uptake by Gender
        # =======================
        st.subheader("📈 Vaccine Uptake by Gender")
        gender_vacc = final_df.groupby('sex')[['h1n1_vaccine','seasonal_vaccine']].mean().reset_index()
        gender_vacc[['h1n1_vaccine','seasonal_vaccine']] *= 100

        fig4 = px.bar(
            gender_vacc,
            x='sex',
            y=['h1n1_vaccine','seasonal_vaccine'],
            barmode='group',
            labels={'value': 'Uptake Rate (%)', 'variable': 'Vaccine Type'},
            text_auto='.2f'
        )
        fig4.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
        fig4.update_layout(yaxis_title="Uptake Rate (%)")
        st.plotly_chart(fig4)
        st.markdown("---")


    if selection == "Demographic Insights":
        st.header("👥 Demographic Insights")

        # ------------------------
        # Sidebar Age Group Filter
        # ------------------------
        st.sidebar.title("🔍 Filter by Age Group")
        
        age_group_options = df['age_group'].dropna().sort_values().unique()
        selected_age = st.sidebar.multiselect("Age Group", age_group_options)

        # ------------------------
        # Apply Age Group Filter
        # ------------------------
        final_df = df.copy()
        if selected_age:
            final_df = final_df[final_df["age_group"].isin(selected_age)]

        # ------------------------
        # Vaccine Uptake By Employment Status
        # ------------------------
        st.subheader("💼 Vaccine Uptake by Employment Status")
        emp_vacc = final_df.groupby('employment_status')[['h1n1_vaccine','seasonal_vaccine']].mean().reset_index()
        emp_vacc[['h1n1_vaccine','seasonal_vaccine']] *= 100
        fig5 = px.bar(
            emp_vacc,
            x='employment_status',
            y=['h1n1_vaccine','seasonal_vaccine'],
            barmode='group',
            labels={'value': 'Uptake Rate (%)', 'variable': 'Vaccine Type'},
            text_auto='.2f'
        )
        fig5.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
        fig5.update_layout(yaxis_title="Uptake Rate (%)")
        st.plotly_chart(fig5)
        st.markdown("---")

        # ------------------------
        # Vaccine Uptake By Marital Status
        # ------------------------
        st.subheader("💍 Vaccine Uptake by Marital Status")
        mar_vacc = final_df.groupby('marital_status')[['h1n1_vaccine','seasonal_vaccine']].mean().reset_index()
        mar_vacc[['h1n1_vaccine','seasonal_vaccine']] *= 100
        fig6 = px.bar(
            mar_vacc,
            x='marital_status',
            y=['h1n1_vaccine','seasonal_vaccine'],
            barmode='group',
            labels={'value': 'Uptake Rate (%)', 'variable': 'Vaccine Type'},
            text_auto='.2f'
        )
        fig6.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
        fig6.update_layout(yaxis_title="Uptake Rate (%)")
        st.plotly_chart(fig6)

        # ------------------------
        # Vaccine Uptake by Race
        # ------------------------
        st.subheader("🏙️ Vaccine Uptake by Race")
        race_vacc = final_df.groupby('race')[['h1n1_vaccine','seasonal_vaccine']].mean().reset_index()
        race_vacc[['h1n1_vaccine','seasonal_vaccine']] *= 100
        fig7 = px.bar(
            race_vacc,
            x='race',
            y=['h1n1_vaccine','seasonal_vaccine'],
            barmode='group',
            labels={'value': 'Uptake Rate (%)', 'variable': 'Vaccine Type'},
            text_auto='.2f'
        )
        fig7.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
        fig7.update_layout(yaxis_title="Uptake Rate (%)")
        st.plotly_chart(fig7)

        # ------------------------
        # Vaccine Uptake by Region
        # ------------------------
        st.subheader("🏙️ Vaccine Uptake by Region")
        region_lookup = {
            "lzgpxyit": "Boston (New England)",
            "fpwskwrf": "New York",
            "mlyzmhmf": "Philadelphia",
            "bhuqouqj": "Atlanta",
            "oxchjgsf": "Chicago",
            "atmpeygn": "Dallas",
            "dqpwygqj": "Kansas City",
            "kbazzjca": "Denver",
            "qufhixun": "San Francisco",
            "lrircsnp": "Seattle"
        }
        final_df['region_name'] = final_df['hhs_geo_region'].map(region_lookup).fillna(final_df['hhs_geo_region'])

        region_vacc = final_df.groupby('region_name')[['h1n1_vaccine','seasonal_vaccine']].mean().reset_index()
        region_vacc[['h1n1_vaccine','seasonal_vaccine']] *= 100
        region_order = [
            "Boston (New England)", "New York", "Philadelphia", "Atlanta", "Chicago",
            "Dallas", "Kansas City", "Denver", "San Francisco", "Seattle"
        ]
        region_vacc['region_name'] = pd.Categorical(region_vacc['region_name'], categories=region_order, ordered=True)
        region_vacc = region_vacc.sort_values('region_name')

        fig8 = px.bar(
            region_vacc,
            x='region_name',
            y=['h1n1_vaccine','seasonal_vaccine'],
            barmode='group',
            labels={'value': 'Uptake Rate (%)', 'variable': 'Vaccine Type', 'region_name': 'Region'},
            text_auto='.2f'
        )
        fig8.update_traces(texttemplate='%{y:.2f}%', textposition='outside')
        fig8.update_layout(yaxis_title="Uptake Rate (%)")
        st.plotly_chart(fig8)



    # ------------------------
    # Additional Visualizations
    # ------------------------
    if selection == "Additional Visualizations":
        st.header("📊 Additional Visualizations")
        # 🔍 Correlation Heatmap
        st.subheader("🔍 Correlation Heatmap")
        plt.figure(figsize=(10,6))
        numeric_df = df.select_dtypes(include='number')  # select only numeric columns
        sns.heatmap(numeric_df.corr(), cmap="coolwarm", center=0, annot=True, fmt=".2f")
        st.pyplot(plt.gcf())  # use gcf() to get current figure
        plt.clf()  # clear the figure for next plot

        # 📊 Vaccination Rates by Concern & Knowledge
        st.subheader("📊 Vaccination Rates by Concern & Knowledge")
        grouped = df.groupby(['h1n1_concern', 'h1n1_knowledge'])['h1n1_vaccine'].mean().reset_index()
        pivot_table = grouped.pivot(index='h1n1_concern', columns='h1n1_knowledge', values='h1n1_vaccine')
        plt.figure(figsize=(8,6))
        sns.heatmap(pivot_table, annot=True, fmt=".2f", cmap="YlGnBu")
        plt.title("H1N1 Vaccination Rates by Concern & Knowledge")
        st.pyplot(plt.gcf())
        plt.clf()

        st.markdown("---")

        

# ------------------------
# Run app
# ------------------------
if not st.session_state["authenticated"]:
    login()
else:
    main_app()
