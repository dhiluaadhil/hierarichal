import streamlit as st
import joblib
import numpy as np
import os

st.set_page_config(page_title="Customer Segmenter", page_icon="🛍️")
st.title("🛍️ Customer Loyalty Segmentation (Hierarchical)")

model_path = 'retail_h_classifier.pkl'
scaler_path = 'retail_scaler.pkl'

if os.path.exists(model_path) and os.path.exists(scaler_path):
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    st.write("Enter customer RFM metrics to determine their segment.")

    # User Inputs
    recency = st.number_input("Recency (Days since last purchase)", 0, 400, 30)
    frequency = st.number_input("Frequency (Total number of orders)", 1, 100, 5)
    monetary = st.number_input("Monetary (Total $ spent)", 0.0, 50000.0, 500.0)

    if st.button("Analyze Customer"):
        # Features: [Recency, Frequency, Monetary]
        raw_data = np.array([[recency, frequency, monetary]])
        scaled_data = scaler.transform(raw_data)
        segment = model.predict(scaled_data)
        
        # Mapping segments to labels
        labels = {0: "🥉 Occasional Shopper", 1: "🥈 Frequent Buyer", 2: "🥇 VIP/Loyal Customer"}
        st.success(f"### Segment: **{labels.get(segment[0], 'General')}**")
        st.info("Use this data to target customers with specific marketing campaigns.")
else:
    st.error("Missing model or scaler files. Please upload them to GitHub.")