# traffic_predictor_app.py
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model #type:ignore
from sklearn.preprocessing import MinMaxScaler

# Load the trained LSTM model
model = load_model('traffic_flow_lstm_model.h5')

# Load and preprocess data
data = pd.read_csv('complete_dataset.csv')  # Adjust the path to your dataset

# Initialize the MinMaxScaler (if you used scaling during training)
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(data[['Traffic Volume']])  # Fit scaler on the complete data (or training data only)

# Function to create input features for prediction
def prepare_input(city, date, time, look_back=3):
    # Filter data for the selected city, date, and time
    filtered_data = data[(data['Location'] == city) & (data['Date'] <= date)]
    if len(filtered_data) < look_back:
        st.warning("Not enough data to make a prediction for the selected inputs.")
        return None, None
    
    filtered_data = filtered_data[-look_back:]  # Take the last 'look_back' rows
    feature_data = filtered_data[['Traffic Volume', 'Temperature']].values
    feature_data = scaler.transform(feature_data)  # Scale the data
    feature_data = np.expand_dims(feature_data, axis=0)  # Reshape for LSTM input
    return feature_data, filtered_data.iloc[-1]  # Return last row for display of parameters
'''
# Streamlit UI setup
st.title('Traffic Volume Predictor')

# User input for City, Date, and Time
city = st.selectbox('Select City', data['Location'].unique())
date = st.date_input('Select Date')
time = st.time_input('Select Time')

if st.button('Predict Traffic Volume'):
    X_input, last_row = prepare_input(city, str(date), str(time))
    if X_input is not None:
        # Predict traffic volume
        predicted_volume_scaled = model.predict(X_input)
        predicted_volume = scaler.inverse_transform(predicted_volume_scaled)[0, 0]
        
        # Display predicted traffic volume
        st.write(f"**Predicted Traffic Volume:** {predicted_volume:.2f}")
        
        # Display additional parameters as caption
        st.caption(f"Temperature: {last_row['Temperature']}°C | Visibility: {last_row['Visibility']} km | Other parameters...")
    else:
        st.error("Could not prepare data for prediction. Check your input or data availability.")
'''