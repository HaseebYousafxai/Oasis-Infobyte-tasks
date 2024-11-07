import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="🚗",
    layout="wide"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .title {
        color: #2E86C1;
        text-align: center;
        font-size: 3em;
        margin-bottom: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #2E86C1;
        color: white;
        padding: 0.5rem;
    }
    .prediction-box {
        background-color: #F7F9F9;
        padding: 2rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-top: 2rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Load the model and scaler
@st.cache_resource
def load_model():
    model = pickle.load(open('linear_model.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))
    return model, scaler

model, scaler = load_model()

# Title
st.markdown("<h1 class='title'>🚗 Car Price Predictor</h1>", unsafe_allow_html=True)

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.subheader("Car Details")
    year = st.slider("Manufacturing Year", 2000, 2024, 2020)
    present_price = st.number_input("Present Market Price (in Lakhs)", 1.0, 50.0, 5.0)
    driven_kms = st.number_input("Kilometers Driven", 0, 500000, 50000)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG"])
    owner = st.selectbox("Number of Previous Owners", [0, 1, 2, 3])

with col2:
    st.subheader("Additional Features")
    selling_type = st.selectbox("Selling Type", ["Individual", "Dealer"])
    transmission = st.selectbox("Transmission Type", ["Manual", "Automatic"])

# Create a dictionary for fuel type mapping
fuel_type_map = {"Petrol": 0, "Diesel": 1, "CNG": 2}

if st.button("Predict Price", key="predict_button"):
    try:
        # Calculate car age
        car_age = 2024 - year
        
        # Create input dataframe with exact column names and order
        input_data = pd.DataFrame({
            'Year': [year],
            'Present_Price': [present_price],
            'Driven_kms': [driven_kms],
            'Fuel_Type': [fuel_type_map[fuel_type]],
            'Owner': [owner],
            'car_age': [car_age],
            'Selling_type_Dealer': [1 if selling_type == "Dealer" else 0],
            'Selling_type_Individual': [1 if selling_type == "Individual" else 0],
            'Transmission_Automatic': [1 if transmission == "Automatic" else 0],
            'Transmission_Manual': [1 if transmission == "Manual" else 0]
        })
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        
        # Display prediction
        st.markdown("<div class='prediction-box'>", unsafe_allow_html=True)
        st.markdown(f"<h2 style='text-align: center; color: #2E86C1;'>Predicted Car Price</h2>", unsafe_allow_html=True)
        st.markdown(f"<h1 style='text-align: center; color: #27AE60;'>₹{prediction:.2f} Lakhs</h1>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
        
        # Create a gauge chart using plotly.graph_objects
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = prediction,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Price Range (in Lakhs)"},
            gauge = {
                'axis': {'range': [None, 50]},
                'bar': {'color': "#2E86C1"},
                'steps': [
                    {'range': [0, 10], 'color': 'lightgray'},
                    {'range': [10, 25], 'color': 'gray'},
                    {'range': [25, 50], 'color': 'darkgray'}
                ]
            }
        ))
        
        fig.update_layout(height=400)
        st.plotly_chart(fig)
        
        # Display input summary
        st.subheader("Input Summary:")
        summary_col1, summary_col2 = st.columns(2)
        
        with summary_col1:
            st.write("**Car Details:**")
            st.write(f"- Year: {year}")
            st.write(f"- Present Price: ₹{present_price} Lakhs")
            st.write(f"- Kilometers Driven: {driven_kms:,} km")
            st.write(f"- Fuel Type: {fuel_type}")
            st.write(f"- Owner: {owner}")
            
        with summary_col2:
            st.write("**Additional Features:**")
            st.write(f"- Selling Type: {selling_type}")
            st.write(f"- Transmission: {transmission}")
            st.write(f"- Car Age: {car_age} years")
        
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check your input values and try again.")

# Add footer with developer info
st.markdown("---")
st.markdown("""
<div style='text-align: center;'>
    <h4>Developed by Haseeb Ahmad</h4>
    <p>Connect with me:</p>
    <a href="https://github.com/HaseebYousafxai/" target="_blank">GitHub</a> |
    <a href="https://www.linkedin.com/in/haseebahmadiuse/" target="_blank">LinkedIn</a>
</div>
""", unsafe_allow_html=True)