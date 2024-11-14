# app_gb.py
import streamlit as st
import pandas as pd
import joblib

# Load the Gradient Boosting model
gb_model = joblib.load('gb_model.pkl')

# Title and Description
st.title("Diagnosis Prediction with Gradient Boosting")
st.write("This app uses a Gradient Boosting model to predict diagnosis based on user input.")

# User inputs for the features
Age = st.number_input("Age", min_value=0)
BMI = st.number_input("BMI", min_value=0.0)
Sex = st.selectbox("Sex", options=[0, 1])  # 0 = Female, 1 = Male
Height = st.number_input("Height (in cm)", min_value=0)
Weight = st.number_input("Weight (in kg)", min_value=0.0)
Diagnosis = st.selectbox("Diagnosis (Optional for User Input)", options=[0, 1])  # Diagnosis should be predicted, but can include it for reference

Alvarado_Score = st.number_input("Alvarado Score", min_value=0)
Paedriatic_Appendicitis_Score = st.number_input("Paediatric Appendicitis Score", min_value=0)
Migratory_Pain = st.selectbox("Migratory Pain (1 for Yes, 0 for No)", options=[0, 1])
Lower_Right_Abd_Pain = st.selectbox("Lower Right Abdomen Pain (1 for Yes, 0 for No)", options=[0, 1])
Contralateral_Rebound_Tenderness = st.selectbox("Contralateral Rebound Tenderness (1 for Yes, 0 for No)", options=[0, 1])
Coughing_Pain = st.selectbox("Coughing Pain (1 for Yes, 0 for No)", options=[0, 1])
Nausea = st.selectbox("Nausea (1 for Yes, 0 for No)", options=[0, 1])
Loss_of_Appetite = st.selectbox("Loss of Appetite (1 for Yes, 0 for No)", options=[0, 1])
Body_Temperature = st.number_input("Body Temperature (°C)", min_value=0.0)
Dysuria = st.selectbox("Dysuria (1 for Yes, 0 for No)", options=[0, 1])
Stool = st.selectbox("Stool (1 for Abnormal, 0 for Normal)", options=[0, 1])
Peritonitis = st.selectbox("Peritonitis (1 for Yes, 0 for No)", options=[0, 1])
Psoas_Sign = st.selectbox("Psoas Sign (1 for Positive, 0 for Negative)", options=[0, 1])

# Organize input into a single DataFrame row
input_data = pd.DataFrame([[Age, BMI, Sex, Height, Weight, Diagnosis, Alvarado_Score,
                            Paedriatic_Appendicitis_Score, Migratory_Pain, Lower_Right_Abd_Pain,
                            Contralateral_Rebound_Tenderness, Coughing_Pain, Nausea, Loss_of_Appetite,
                            Body_Temperature, Dysuria, Stool, Peritonitis, Psoas_Sign]], 
                          columns=['Age', 'BMI', 'Sex', 'Height', 'Weight', 'Diagnosis', 'Alvarado_Score',
                                   'Paedriatic_Appendicitis_Score', 'Migratory_Pain', 'Lower_Right_Abd_Pain',
                                   'Contralateral_Rebound_Tenderness', 'Coughing_Pain', 'Nausea', 'Loss_of_Appetite',
                                   'Body_Temperature', 'Dysuria', 'Stool', 'Peritonitis', 'Psoas_Sign'])

# Predict button
if st.button("Predict"):
    # Make prediction
    gb_pred = gb_model.predict(input_data)[0]
    gb_accuracy = 0.85  # Placeholder; update based on actual accuracy

    # Display the prediction and accuracy
    st.subheader("Prediction")
    st.write("Gradient Boosting Prediction:", gb_pred)
    st.write("Gradient Boosting Accuracy:", gb_accuracy)
