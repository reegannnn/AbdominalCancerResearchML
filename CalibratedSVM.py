import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the calibrated model
svm_model = joblib.load('calibrated_svm_model.pkl')

# Load data for visualizations
data = pd.read_csv("final_data.csv")

# Title and Description
st.title("Diagnosis Prediction with Calibrated SVM Model")
st.write("This app uses a calibrated Support Vector Machine (SVM) model to predict diagnosis based on user input.")

# Plotting Section (as before)
# Add histograms, scatter plots, and other visuals here as in the code provided earlier.

# User inputs for the features
Age = st.number_input("Age", min_value=0)
BMI = st.number_input("BMI", min_value=0.0)
Sex = st.selectbox("Sex", options=[0, 1])  # 0 = Female, 1 = Male
Height = st.number_input("Height (in cm)", min_value=0)
Weight = st.number_input("Weight (in kg)", min_value=0.0)
Diagnosis = st.selectbox("Diagnosis (Optional for User Input)", options=[0, 1])

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

# Ensure input columns align with model's expected features
input_data = input_data[svm_model.feature_names_in_]

# Predict using adjusted threshold
if st.button("Predict"):
    prediction_proba = svm_model.predict_proba(input_data)[0][1]
    threshold = 0.4  # Adjust threshold as needed
    prediction = 1 if prediction_proba >= threshold else 0

    # Display prediction results
    st.subheader("Prediction with Adjusted Threshold")
    st.write("Diagnosis Prediction (0 = No, 1 = Yes):", prediction)
    st.write("Probability of Diagnosis being positive (Yes):", round(prediction_proba, 4))

    # Placeholder for accuracy if you wish to display it
    svm_accuracy = 0.85  # Update this based on model evaluation
    st.write("SVM Model Accuracy:", svm_accuracy)
