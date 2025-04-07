import streamlit as st
import pandas as pd
import numpy as np
import joblib
from geopy.distance import geodesic

# Load trained models and encoders
rf_model = joblib.load("models/random_forest_model1.pkl")
xgb_model = joblib.load("models/xgboost_model1.pkl")
label_encoders = joblib.load("models/label_encoders1.pkl")
onehot_encoder = joblib.load("models/onehot_encoder1.pkl")
feature_order = joblib.load("models/feature_order1.pkl")

# Define function for distance calculation
def haversine(lat1, lon1, lat2, lon2):
    return geodesic((lat1, lon1), (lat2, lon2)).km

# Streamlit App
def main():
    st.title("Credit Card Fraud Detection System")
    st.write("Enter transaction details to predict if it's fraudulent.")
    
    # User input fields
    merchant = st.text_input("Merchant Name")
    category = st.text_input("Category")
    amt = st.number_input("Transaction Amount ($)", min_value=0.01, step=0.01)
    cc_num = st.text_input("Credit Card Number")
    hour = st.slider("Transaction Hour", 0, 23, 12)
    day = st.slider("Transaction Day", 1, 31, 15)
    month = st.slider("Transaction Month", 1, 12, 6)
    gender = st.selectbox("Gender", ["Male", "Female"])
    
    # Latitude & Longitude Inputs
    lat = st.number_input("Cardholder Latitude", format="%f")
    lon = st.number_input("Cardholder Longitude", format="%f")
    merch_lat = st.number_input("Merchant Latitude", format="%f")
    merch_long = st.number_input("Merchant Longitude", format="%f")
    
    # Compute distance
    distance = haversine(lat, lon, merch_lat, merch_long)

    # Label Encode Categorical Inputs
    def encode_feature(feature_name, value):
        if value in label_encoders[feature_name].classes_:
            return label_encoders[feature_name].transform([value])[0]
        else:
            return -1  # Assign -1 if unseen value

    encoded_merchant = encode_feature("merchant", merchant)
    encoded_gender = encode_feature("gender", gender)

    # One-Hot Encode 'category'
    category_array = onehot_encoder.transform([[category]])
    category_df = pd.DataFrame(category_array, columns=onehot_encoder.get_feature_names_out(['category']))
    
    # Prepare final input data
    input_data = pd.DataFrame({
        "merchant": [encoded_merchant], 
        "amt": [amt],
        "hour": [hour], 
        "day": [day],
        "month": [month], 
        "gender": [encoded_gender], 
        "distance": [distance]
    })

    # Merge with one-hot encoded category
    input_data = pd.concat([input_data, category_df], axis=1)

    # Ensure feature order matches training
    for col in feature_order:
        if col not in input_data:
            input_data[col] = 0  # Add missing category columns
    input_data = input_data[feature_order]  # Reorder columns

    # Prediction button
    if st.button("Predict Fraud"): 
        rf_pred = rf_model.predict(input_data)[0]
        xgb_pred = xgb_model.predict(input_data)[0]
        
        st.write("### Predictions:")
        st.write(f"**Random Forest:** {'Fraud' if rf_pred == 1 else 'Legitimate'}")
        st.write(f"**XGBoost:** {'Fraud' if xgb_pred == 1 else 'Legitimate'}")
        
        if rf_pred == 1 or xgb_pred == 1:
            st.error("⚠️ Warning! This transaction is flagged as fraudulent.")
        else:
            st.success("✅ This transaction appears legitimate.")

if __name__ == "__main__":
    main()
