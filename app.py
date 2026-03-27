import streamlit as st
import joblib
import numpy as np
import tensorflow as tf

# Load scalers and models
scaler_rf = joblib.load('finance_scaler_rf.pkl')
scaler_price = joblib.load('finance_price_scaler.pkl')
model_lr = joblib.load('finance_lr_model.pkl')
model_rf = joblib.load('finance_rf_model.pkl')
model_lstm = tf.keras.models.load_model('finance_lstm_model.h5')

st.title("Financial Market Predictor")

# Sidebar for model selection
model_choice = st.sidebar.selectbox("Select Model", ["Linear Regression", "Random Forest", "LSTM"])

if model_choice in ["Linear Regression", "Random Forest"]:
    st.subheader(f"Predict using {model_choice}")
    
    # Feature inputs based on your scaler's requirements
    rsi = st.number_input("RSI", value=50.0)
    macd = st.number_input("MACD", value=0.0)
    macd_signal = st.number_input("MACD Signal", value=0.0)
    close = st.number_input("Close Price", value=150.0)
    # ... Add other features (Open, High, Low, Volume, etc.) ...

    if st.button("Predict"):
        # Create input array (must match the 13 features in order)
        input_features = np.array([[rsi, macd, macd_signal, 0, 0, 0, 0, 0, 0, 0, 0, close, 0]])
        scaled_input = scaler_rf.transform(input_features)
        
        if model_choice == "Linear Regression":
            prediction = model_lr.predict(scaled_input)
        else:
            prediction = model_rf.predict(scaled_input)
        
        st.success(f"Prediction: {prediction[0]}")

elif model_choice == "LSTM":
    st.subheader("LSTM Time Series Prediction")
    st.write("This model requires the last 60 days of scaled closing prices.")
    
    # In a real app, you would fetch recent data via an API (like yfinance)
    # Here we use a placeholder for demonstration
    placeholder_data = st.text_area("Enter last 60 scaled prices (comma separated)", "0.5, 0.51, ...")
    
    if st.button("Predict Next Price"):
        # Reshape to (1, 60, 1) as required by your model's Input Layer
        input_data = np.random.rand(1, 60, 1) 
        prediction = model_lstm.predict(input_data)
        
        # Inverse transform to get actual price
        actual_price = scaler_price.inverse_transform(prediction)
        st.success(f"Predicted Next Price: ${actual_price[0][0]:.2f}")
