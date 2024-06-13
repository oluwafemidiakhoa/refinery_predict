import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sqlalchemy import create_engine
from dotenv import load_dotenv
import os
import plotly.express as px
from sklearn.metrics import classification_report, confusion_matrix

# Load environment variables
load_dotenv()
database_url = os.getenv('DATABASE_URL')

# Check if the DATABASE_URL was loaded correctly
if not database_url:
    st.error("DATABASE_URL not set. Please check your .env file.")
    st.stop()

# Connect to the database
engine = create_engine(database_url)

# Load data without caching
def load_data():
    query = "SELECT * FROM predictivemaintenance"
    data = pd.read_sql(query, engine)
    return data

data = load_data()

# Check if the data is empty
if data.empty:
    st.error("No data found in the database table 'predictivemaintenance'. Please ensure the table contains data.")
else:
    # Check if the target variable has both classes
    if len(data['failure'].unique()) < 2:
        st.error("The target variable 'failure' does not contain both classes. Please ensure the data includes both 0 and 1 values.")
        st.stop()

    # Train a model
    def train_model(data):
        X = data.drop(['id', 'failure'], axis=1)
        y = data['failure']
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X, y)
        return model

    model = train_model(data)

    # Make predictions
    def predict_failures(data, model):
        X = data.drop(['id', 'failure'], axis=1)
        data['predicted_failure'] = model.predict(X)
        if len(model.classes_) == 2:
            data['prediction_proba'] = model.predict_proba(X)[:, 1]
        else:
            data['prediction_proba'] = 0.0
        return data

    predictions = predict_failures(data, model)

    # Streamlit app layout
    st.title("RefineryPredict: AI-Driven Predictive Maintenance")
    st.markdown("""
        Upload your equipment data and get predictions on potential failures.
        This application leverages machine learning to predict equipment failures,
        helping you to take preventive actions and minimize downtime.
    """)

    # Upload data
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    if uploaded_file is not None:
        user_data = pd.read_csv(uploaded_file)
        user_predictions = predict_failures(user_data, model)
        
        st.subheader("Uploaded Data")
        st.dataframe(user_data)

        st.subheader("Predicted Failures")
        st.dataframe(user_predictions[['predicted_failure', 'prediction_proba']])

        # Classification Report and Confusion Matrix
        if 'failure' in user_data.columns:
            st.subheader("Model Evaluation")
            y_true = user_data['failure']
            y_pred = user_predictions['predicted_failure']
            report = classification_report(y_true, y_pred, output_dict=True)
            cm = confusion_matrix(y_true, y_pred)
            
            st.write("Classification Report:")
            st.json(report)
            
            st.write("Confusion Matrix:")
            fig_cm = px.imshow(cm, text_auto=True, color_continuous_scale='Blues')
            st.plotly_chart(fig_cm)

    # Sidebar - User Inputs for Feature Selection
    st.sidebar.header("User Input Features")
    selected_feature = st.sidebar.selectbox('Select Feature to Visualize', data.columns)

    # Data Visualization
    st.subheader("Data Visualization")
    st.write(f"Distribution of {selected_feature}")
    fig = px.histogram(data, x=selected_feature, color='failure', barmode='overlay', title=f'Distribution of {selected_feature} by Failure')
    st.plotly_chart(fig)

    st.write("Failure Predictions Visualization")
    fig_predictions = px.histogram(predictions, x='predicted_failure', title='Failure Predictions')
    st.plotly_chart(fig_predictions)

    st.subheader("Prediction Probability Distribution")
    fig_proba = px.histogram(predictions, x='prediction_proba', nbins=50, title='Prediction Probability Distribution')
    st.plotly_chart(fig_proba)
