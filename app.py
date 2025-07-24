# app.py
import streamlit as st
import numpy as np
import joblib

# Load trained model
model = joblib.load("diabetes_model.pkl")

st.title("🩺 Personalized Health Risk Prediction (Diabetes)")
st.write("Enter the values below to check your diabetes risk.")

# Input fields
pregnancies = st.number_input("🤰 Pregnancies", min_value=0)
glucose = st.number_input("🩸 Glucose Level", min_value=0)
bp = st.number_input("💓 Blood Pressure", min_value=0)
skin_thickness = st.number_input("🧍‍♂️ Skin Thickness", min_value=0)
insulin = st.number_input("💉 Insulin Level", min_value=0)
bmi = st.number_input("⚖️ BMI", min_value=0.0)
dpf = st.number_input("🧬 Diabetes Pedigree Function", min_value=0.0)
age = st.number_input("🎂 Age", min_value=1)

# Prediction
if st.button("Predict"):
    user_data = np.array([[pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]])
    prediction = model.predict(user_data)[0]
    if prediction == 1:
        st.error("⚠️ High Risk: You may be at risk of Diabetes.")
    else:
        st.success("✅ Low Risk: You are unlikely to have Diabetes.")
