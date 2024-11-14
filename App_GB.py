# app_gb.py
import streamlit as st
import pandas as pd
import joblib

# Load the Gradient Boosting model
gb_model = joblib.load('gb_model.pkl')

# Title and Description
st.title("Diagnosis Prediction with Gradient Boosting")
st.write("This app uses a Gradient Boosting model to predict diagnosis based on user input.")

# User inputs
Alvarado_Score = st.number_input("Alvarado Score")
Paedriatic_Appendicitis_Score = st.number_input("Paediatric Appendicitis Score")
WBC_Count = st.number_input("WBC Count")
Neutrophilia = st.number_input("Neutrophilia")
Management = st.number_input("Management")
Severity = st.number_input("Severity")
Length_of_Stay = st.number_input("Length of Stay")
CRP = st.number_input("CRP")
Peritonitis = st.number_input("Peritonitis")
Age = st.number_input("Age")
BMI = st.number_input("BMI")
Height = st.number_input("Height")
Weight = st.number_input("Weight")
Sex = st.selectbox("Sex", options=[0, 1])  # Assuming 0 = Female, 1 = Male

# Organize input into a single DataFrame row
input_data = pd.DataFrame([[Alvarado_Score, Paedriatic_Appendicitis_Score, WBC_Count, Neutrophilia,
                            Management, Severity, Length_of_Stay, CRP, Peritonitis, Age,
                            BMI, Height, Weight, Sex]], 
                          columns=['Alvarado_Score', 'Paedriatic_Appendicitis_Score', 'WBC_Count', 
                                   'Neutrophilia', 'Management', 'Severity', 'Length_of_Stay', 
                                   'CRP', 'Peritonitis', 'Age', 'BMI', 'Height', 'Weight', 'Sex'])

# Predict button
if st.button("Predict"):
    # Make prediction
    gb_pred = gb_model.predict(input_data)[0]
    gb_accuracy = 0.85  # Placeholder; update based on actual accuracy

    # Display the prediction and accuracy
    st.subheader("Prediction")
    st.write("Gradient Boosting Prediction:", gb_pred)
    st.write("Gradient Boosting Accuracy:", gb_accuracy)
