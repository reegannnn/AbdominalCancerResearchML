# import streamlit as st
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the model
# @st.cache_resource
# def load_model():
#     model = joblib.load("diagnostic_model.pkl")
#     return model

# # Load data
# @st.cache_data
# def load_data():
#     data = pd.read_csv("app_data.csv")
    
#     # Derive necessary columns if they don’t exist directly
#     data["Management_primary surgical"] = (data["Management"] == "primary surgical").astype(int)
#     data["Appendix_on_US_yes"] = (data["Appendix_on_US"] == "yes").astype(int)
#     data["Diagnosis_Presumptive_no appendicitis"] = (data["Diagnosis_Presumptive"] == "no appendicitis").astype(int)
#     data["Severity_uncomplicated"] = (data["Severity"] == "uncomplicated").astype(int)
    
#     return data

# # Main app function
# def main():
#     st.title("Cancer Research Data Analysis and Prediction")
#     st.write("This app allows you to explore cancer research data and make predictions based on key features.")

#     # Load and display data
#     data = load_data()
#     model = load_model()

#     # Selecting specified columns
#     features = [
#         'Appendix_Diameter', 'Management_primary surgical', 'Appendix_on_US_yes',
#         'US_Number', 'Length_of_Stay', 'WBC_Count', 'CRP', 
#         'Diagnosis_Presumptive_no appendicitis', 'Alvarado_Score', 'Severity_uncomplicated',
#         'Age', 'Weight', 'Height', 'BMI', 'Sex'
#     ]

#     # Filter dataset for the selected features
#     data = data[features]
#     st.write("### Data Preview (Selected Columns)")
#     st.write(data.head())

#     # Sidebar filters
#     st.sidebar.header("Filter Options")
#     if 'Age' in data.columns:
#         min_age, max_age = int(data['Age'].min()), int(data['Age'].max())
#         age_filter = st.sidebar.slider("Filter by Age", min_value=min_age, max_value=max_age, value=(min_age, max_age))
#         data = data[(data['Age'] >= age_filter[0]) & (data['Age'] <= age_filter[1])]

#     # Visualization
#     st.write("## Data Visualizations")

#     # Histogram for Appendix Diameter
#     if 'Appendix_Diameter' in data.columns:
#         st.write("### Histogram: Appendix Diameter")
#         fig, ax = plt.subplots()
#         sns.histplot(data['Appendix_Diameter'], kde=True, ax=ax)
#         st.pyplot(fig)

#     # Scatter plot for Age vs BMI
#     if 'Age' in data.columns and 'BMI' in data.columns:
#         st.write("### Scatter Plot: Age vs BMI")
#         fig, ax = plt.subplots()
#         sns.scatterplot(data=data, x='Age', y='BMI', hue='Sex', ax=ax)
#         st.pyplot(fig)

#     # Prediction Section
#     st.write("## Predictive Analysis")
#     st.write("Enter values below to get a prediction.")

#     # Input form for prediction
#     with st.form("prediction_form"):
#         input_values = {}
#         for feature in features:
#             if feature in ['Sex', 'Management_primary surgical', 'Appendix_on_US_yes', 'Diagnosis_Presumptive_no appendicitis', 'Severity_uncomplicated']:
#                 input_values[feature] = st.selectbox(f"{feature}", [0, 1])
#             else:
#                 input_values[feature] = st.number_input(f"{feature}", min_value=0.0)

#         # Submit button for prediction
#         submit = st.form_submit_button("Predict")

#     if submit:
#         # Convert input values to DataFrame
#         input_df = pd.DataFrame([input_values])

#         # Make a prediction
#         prediction = model.predict(input_df)[0]
#         prediction_proba = model.predict_proba(input_df)[0]

#         # Display prediction
#         st.write(f"### Prediction: {'Appendicitis' if prediction == 1 else 'No Appendicitis'}")
#         st.write(f"Probability of Appendicitis: {prediction_proba[1]:.2f}")
#         st.write(f"Probability of No Appendicitis: {prediction_proba[0]:.2f}")

# # Run the app
# if __name__ == "__main__":
#     main()

# import streamlit as st
# import pandas as pd
# import joblib
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the model
# @st.cache_resource
# def load_model():
#     model = joblib.load("diagnostic_model.pkl")
#     return model

# # Load data
# @st.cache_data
# def load_data():
#     data = pd.read_csv("app_data.csv")
    
#     # Derive necessary columns if they don’t exist directly
#     data["Management_primary surgical"] = (data["Management"] == "primary surgical").astype(int)
#     data["Appendix_on_US_yes"] = (data["Appendix_on_US"] == "yes").astype(int)
#     data["Diagnosis_Presumptive_no appendicitis"] = (data["Diagnosis_Presumptive"] == "no appendicitis").astype(int)
#     data["Severity_uncomplicated"] = (data["Severity"] == "uncomplicated").astype(int)
    
#     return data

# # Main app function
# def main():
#     st.title("Cancer Research Data Analysis and Prediction")
#     st.write("This app allows you to explore cancer research data and make predictions based on key features.")

#     # Load and display data
#     data = load_data()
#     model = load_model()

#     # Define feature names in the correct order expected by the model
#     features = [
#         'Appendix_Diameter', 'Management_primary surgical', 'Appendix_on_US_yes',
#         'US_Number', 'Length_of_Stay', 'WBC_Count', 'CRP', 
#         'Diagnosis_Presumptive_no appendicitis', 'Alvarado_Score', 'Severity_uncomplicated',
#         'Age', 'Weight', 'Height', 'BMI', 'Sex'
#     ]

#     # Filter dataset for the selected features
#     data = data[features]
#     st.write("### Data Preview (Selected Columns)")
#     st.write(data.head())

#     # Sidebar filters
#     st.sidebar.header("Filter Options")
#     if 'Age' in data.columns:
#         min_age, max_age = int(data['Age'].min()), int(data['Age'].max())
#         age_filter = st.sidebar.slider("Filter by Age", min_value=min_age, max_value=max_age, value=(min_age, max_age))
#         data = data[(data['Age'] >= age_filter[0]) & (data['Age'] <= age_filter[1])]

#     # Visualization
#     st.write("## Data Visualizations")

#     # Histogram for Appendix Diameter
#     if 'Appendix_Diameter' in data.columns:
#         st.write("### Histogram: Appendix Diameter")
#         fig, ax = plt.subplots()
#         sns.histplot(data['Appendix_Diameter'], kde=True, ax=ax)
#         st.pyplot(fig)

#     # Scatter plot for Age vs BMI
#     if 'Age' in data.columns and 'BMI' in data.columns:
#         st.write("### Scatter Plot: Age vs BMI")
#         fig, ax = plt.subplots()
#         sns.scatterplot(data=data, x='Age', y='BMI', hue='Sex', ax=ax)
#         st.pyplot(fig)

#     # Prediction Section
#     st.write("## Predictive Analysis")
#     st.write("Enter values below to get a prediction.")

#     # Input form for prediction
#     with st.form("prediction_form"):
#         input_values = {}
#         for feature in features:
#             if feature in ['Sex', 'Management_primary surgical', 'Appendix_on_US_yes', 'Diagnosis_Presumptive_no appendicitis', 'Severity_uncomplicated']:
#                 input_values[feature] = st.selectbox(f"{feature}", [0, 1])
#             else:
#                 input_values[feature] = st.number_input(f"{feature}", min_value=0.0)

#         # Submit button for prediction
#         submit = st.form_submit_button("Predict")

#     if submit:
#         # Convert input values to DataFrame and ensure correct feature order
#         input_df = pd.DataFrame([input_values], columns=features)

#         # Make a prediction
#         prediction = model.predict(input_df)[0]
#         prediction_proba = model.predict_proba(input_df)[0]

#         # Display prediction
#         st.write(f"### Prediction: {'Appendicitis' if prediction == 1 else 'No Appendicitis'}")
#         st.write(f"Probability of Appendicitis: {prediction_proba[1]:.2f}")
#         st.write(f"Probability of No Appendicitis: {prediction_proba[0]:.2f}")

# # Run the app
# if __name__ == "__main__":
#     main()

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

# Load data
@st.cache_data
def load_data():
    data = pd.read_csv("app_data.csv")
    
    # Derive necessary columns if they don’t exist directly
    data["Management_primary surgical"] = (data["Management"] == "primary surgical").astype(int)
    data["Appendix_on_US_yes"] = (data["Appendix_on_US"] == "yes").astype(int)
    data["Diagnosis_Presumptive_no appendicitis"] = (data["Diagnosis_Presumptive"] == "no appendicitis").astype(int)
    data["Severity_uncomplicated"] = (data["Severity"] == "uncomplicated").astype(int)
    
    return data

# Main app function
def main():
    st.title("Cancer Research Data Analysis and Prediction")
    st.write("This app allows you to explore cancer research data and make predictions based on key features.")

    # Load and display data
    data = load_data()
    model = load_model()

    # Define feature names in the correct order expected by the model
    features = [
        'Appendix_Diameter', 'Management_primary surgical', 'Appendix_on_US_yes',
        'US_Number', 'Length_of_Stay', 'WBC_Count', 'CRP', 
        'Diagnosis_Presumptive_no appendicitis', 'Alvarado_Score', 'Severity_uncomplicated',
        'Age', 'Weight', 'Height', 'BMI', 'Sex'
    ]

    # Filter dataset for the selected features
    data = data[features]
    st.write("### Data Preview (Selected Columns)")
    st.write(data.head())

    # Sidebar filters
    st.sidebar.header("Filter Options")
    
    # Age filter
    if 'Age' in data.columns:
        min_age, max_age = int(data['Age'].min()), int(data['Age'].max())
        age_filter = st.sidebar.slider("Filter by Age", min_value=min_age, max_value=max_age, value=(min_age, max_age))
        data = data[(data['Age'] >= age_filter[0]) & (data['Age'] <= age_filter[1])]

    # BMI filter
    if 'BMI' in data.columns:
        min_bmi, max_bmi = float(data['BMI'].min()), float(data['BMI'].max())
        bmi_filter = st.sidebar.slider("Filter by BMI", min_value=min_bmi, max_value=max_bmi, value=(min_bmi, max_bmi))
        data = data[(data['BMI'] >= bmi_filter[0]) & (data['BMI'] <= bmi_filter[1])]

    # Height filter
    if 'Height' in data.columns:
        min_height, max_height = float(data['Height'].min()), float(data['Height'].max())
        height_filter = st.sidebar.slider("Filter by Height (cm)", min_value=min_height, max_value=max_height, value=(min_height, max_height))
        data = data[(data['Height'] >= height_filter[0]) & (data['Height'] <= height_filter[1])]

    # Weight filter
    if 'Weight' in data.columns:
        min_weight, max_weight = float(data['Weight'].min()), float(data['Weight'].max())
        weight_filter = st.sidebar.slider("Filter by Weight (kg)", min_value=min_weight, max_value=max_weight, value=(min_weight, max_weight))
        data = data[(data['Weight'] >= weight_filter[0]) & (data['Weight'] <= weight_filter[1])]

    # Sex filter
    # if 'Sex' in data.columns:
    #     sex_filter = st.sidebar.selectbox("Filter by Sex", options=[0,1])
    #     if sex_filter != "All":
    #         data = data[data['Sex'] == sex_filter]

    # Visualization
    st.write("## Data Visualizations")

    # Histogram for Appendix Diameter
    if 'Appendix_Diameter' in data.columns:
        st.write("### Histogram: Appendix Diameter")
        fig, ax = plt.subplots()
        sns.histplot(data['Appendix_Diameter'], kde=True, ax=ax)
        st.pyplot(fig)

    # Scatter plot for Age vs BMI
    if 'Age' in data.columns and 'BMI' in data.columns:
        st.write("### Scatter Plot: Age vs BMI")
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x='Age', y='BMI', hue='Sex', ax=ax)
        st.pyplot(fig)

    # Prediction Section
    st.write("## Predictive Analysis")
    st.write("Enter values below to get a prediction.")

    # Input form for prediction
    with st.form("prediction_form"):
        input_values = {}
        for feature in features:
            if feature in ['Sex', 'Management_primary surgical', 'Appendix_on_US_yes', 'Diagnosis_Presumptive_no appendicitis', 'Severity_uncomplicated']:
                input_values[feature] = st.selectbox(f"{feature}", [0, 1])
            else:
                input_values[feature] = st.number_input(f"{feature}", min_value=0.0)

        # Submit button for prediction
        submit = st.form_submit_button("Predict")

    if submit:
        # Convert input values to DataFrame and ensure correct feature order
        input_df = pd.DataFrame([input_values], columns=features)

        # Make a prediction
        prediction = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]

        # Display prediction
        st.write(f"### Prediction: {'Appendicitis' if prediction == 1 else 'No Appendicitis'}")
        st.write(f"Probability of Appendicitis: {prediction_proba[1]:.2f}")
        st.write(f"Probability of No Appendicitis: {prediction_proba[0]:.2f}")

# Run the app
if __name__ == "__main__":
    main()
