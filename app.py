import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Load the model
@st.cache_resource
def load_model():
    model = joblib.load("diagnostic_model.pkl")
    return model

# Load the data from CSV
@st.cache_data
def load_data():
    data = pd.read_csv("app_data.csv")
    return data

# Main app function
def main():
    st.title("Cancer Research Data Analysis & Diagnosis")
    st.write("This app allows you to explore and visualize cancer research data and make diagnostic predictions based on key features.")

    # Load model and data
    model = load_model()
    data = load_data()
    
    # Selecting specified columns for analysis
    features = [
        'Appendix_Diameter', 'Management', 'Appendix_on_US_yes', 
        'US_Number', 'Length_of_Stay', 'WBC_Count', 'CRP', 
        'Diagnosis_Presumptive_no appendicitis', 'Alvarado_Score', 
        'Severity_uncomplicated', 'Age', 'Weight', 'Height', 
        'BMI', 'Sex'
    ]
    
    # Filter dataset for selected features
    data = data[features]
    st.write("### Data Preview (Selected Columns)")
    st.write(data.head())

    # Show basic stats for selected features
    st.write("### Basic Statistics for Selected Features")
    st.write(data.describe())

    # Sidebar filters for specific features
    st.sidebar.header("Filter Options")
    
    if 'Age' in data.columns:
        min_age, max_age = int(data['Age'].min()), int(data['Age'].max())
        age_filter = st.sidebar.slider("Filter by Age", min_value=min_age, max_value=max_age, value=(min_age, max_age))
        data = data[(data['Age'] >= age_filter[0]) & (data['Age'] <= age_filter[1])]
    
    if 'BMI' in data.columns:
        min_bmi, max_bmi = int(data['BMI'].min()), int(data['BMI'].max())
        bmi_filter = st.sidebar.slider("Filter by BMI", min_value=min_bmi, max_value=max_bmi, value=(min_bmi, max_bmi))
        data = data[(data['BMI'] >= bmi_filter[0]) & (data['BMI'] <= bmi_filter[1])]

    # Data Visualization Options
    st.write("## Data Visualizations")

    if 'Appendix_Diameter' in data.columns:
        st.write("### Histogram: Appendix Diameter")
        fig, ax = plt.subplots()
        sns.histplot(data['Appendix_Diameter'], kde=True, ax=ax)
        st.pyplot(fig)

    if 'Age' in data.columns and 'BMI' in data.columns:
        st.write("### Scatter Plot: Age vs BMI")
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x='Age', y='BMI', hue='Sex', ax=ax)
        st.pyplot(fig)

    if 'Management' in data.columns:
        st.write("### Count Plot: Management")
        fig, ax = plt.subplots()
        sns.countplot(data=data, x='Management', ax=ax)
        st.pyplot(fig)

    st.write("### Correlation Matrix for Numeric Features")
    numeric_features = data.select_dtypes(include=['float64', 'int64']).dropna()
    if not numeric_features.empty:
        corr = numeric_features.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Prediction Interface
    st.write("## Make a Diagnostic Prediction")

    # Input fields for prediction
    appendix_diameter = st.number_input("Appendix Diameter", value=0.0)
    us_number = st.number_input("US Number", value=0)
    length_of_stay = st.number_input("Length of Stay", value=0)
    wbc_count = st.number_input("WBC Count", value=0.0)
    crp = st.number_input("CRP", value=0.0)
    alvarado_score = st.number_input("Alvarado Score", value=0)
    age = st.number_input("Age", value=0)
    weight = st.number_input("Weight", value=0.0)
    height = st.number_input("Height", value=0.0)
    bmi = st.number_input("BMI", value=0.0)

    # Button for prediction
    if st.button("Predict"):
        # Prepare input for the model
        input_data = pd.DataFrame([[appendix_diameter, us_number, length_of_stay, wbc_count, crp, alvarado_score, age, weight, height, bmi]],
                                  columns=['Appendix_Diameter', 'US_Number', 'Length_of_Stay', 'WBC_Count', 'CRP', 'Alvarado_Score', 'Age', 'Weight', 'Height', 'BMI'])
        # Make prediction
        prediction = model.predict(input_data)
        diagnosis = "Appendicitis" if prediction[0] == 1 else "No Appendicitis"
        st.write(f"### Diagnosis: {diagnosis}")

# Run the app
if __name__ == "__main__":
    main()
