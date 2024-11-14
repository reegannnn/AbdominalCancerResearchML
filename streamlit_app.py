import streamlit as st
import joblib
import pandas as pd

# Load the model
model = joblib.load('final_model.pkl')

# Set up the Streamlit app
st.title("Diagnostic Model Prediction")

# Input fields for the model
age = st.number_input("Age")
bmi = st.number_input("BMI")
sex = st.selectbox("Sex", [0, 1])  # Assuming 0 for female, 1 for male
height = st.number_input("Height")
weight = st.number_input("Weight")
alvarado_score = st.number_input("Alvarado Score")
paediatric_score = st.number_input("Paediatric Appendicitis Score")
migratory_pain = st.selectbox("Migratory Pain", [0, 1])
lower_right_abd_pain = st.selectbox("Lower Right Abd Pain", [0, 1])
contralateral_rebound = st.selectbox("Contralateral Rebound Tenderness", [0, 1])
coughing_pain = st.selectbox("Coughing Pain", [0, 1])
nausea = st.selectbox("Nausea", [0, 1])
loss_of_appetite = st.selectbox("Loss of Appetite", [0, 1])
body_temperature = st.number_input("Body Temperature")
dysuria = st.selectbox("Dysuria", [0, 1])
stool = st.selectbox("Stool", [0, 1])
peritonitis = st.selectbox("Peritonitis", [0, 1])
psoas_sign = st.selectbox("Psoas Sign", [0, 1])

# Make prediction
if st.button("Predict"):
    input_data = pd.DataFrame({
        'Age': [age], 'BMI': [bmi], 'Sex': [sex], 'Height': [height], 'Weight': [weight],
        'Alvarado_Score': [alvarado_score], 'Paedriatic_Appendicitis_Score': [paediatric_score],
        'Migratory_Pain': [migratory_pain], 'Lower_Right_Abd_Pain': [lower_right_abd_pain],
        'Contralateral_Rebound_Tenderness': [contralateral_rebound], 'Coughing_Pain': [coughing_pain],
        'Nausea': [nausea], 'Loss_of_Appetite': [loss_of_appetite], 'Body_Temperature': [body_temperature],
        'Dysuria': [dysuria], 'Stool': [stool], 'Peritonitis': [peritonitis], 'Psoas_Sign': [psoas_sign]
    })
    prediction = model.predict(input_data)[0]
    diagnosis = "No Appendicitis" if prediction == 1 else "Appendicitis"
    st.write("Diagnosis:", diagnosis)
