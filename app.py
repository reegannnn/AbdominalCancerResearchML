# import streamlit as st
# import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

# # Load the data from CSV
# @st.cache_data
# def load_data():
#     data = pd.read_csv("app_data.csv")
#     return data

# # Main app function
# def main():
#     st.title("Cancer Research Data Analysis")
#     st.write("This app allows you to explore and visualize cancer research data.")

#     # Load and display data
#     data = load_data()
#     st.write("### Data Preview")
#     st.write(data.head())

#     # Show basic stats
#     st.write("### Basic Data Statistics")
#     st.write(data.describe())

#     # Sidebar filters (example for numerical filtering)
#     st.sidebar.header("Filter Options")
#     if 'Age' in data.columns:
#         min_age, max_age = int(data['Age'].min()), int(data['Age'].max())
#         age_filter = st.sidebar.slider("Filter by Age", min_value=min_age, max_value=max_age, value=(min_age, max_age))
#         data = data[(data['Age'] >= age_filter[0]) & (data['Age'] <= age_filter[1])]

#     # Visualization options
#     st.write("## Data Visualizations")
    
#     # Count plot for a categorical column (example for 'Diagnosis')
#     if 'Diagnosis' in data.columns:
#         st.write("### Diagnosis Distribution")
#         fig, ax = plt.subplots()
#         sns.countplot(data=data, x='Diagnosis', ax=ax)
#         st.pyplot(fig)

#     # Scatter plot example for numerical columns (example: 'Age' vs 'Tumor Size')
#     if 'Age' in data.columns and 'Tumor_Size' in data.columns:
#         st.write("### Scatter Plot: Age vs Tumor Size")
#         fig, ax = plt.subplots()
#         sns.scatterplot(data=data, x='Age', y='Tumor_Size', ax=ax)
#         st.pyplot(fig)

#     # Correlation heatmap
#     # st.write("### Correlation Matrix")
#     # corr = data.corr()
#     # fig, ax = plt.subplots(figsize=(10, 8))
#     # sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
#     # st.pyplot(fig)
#     # Correlation matrix
#     st.write("### Correlation Matrix")
#     numeric_data = data.select_dtypes(include=['float64', 'int64'])  # Select only numeric columns
#     if numeric_data.empty:
#         st.write("No numeric columns available for correlation.")
#     else:
#         corr = numeric_data.corr()
#         fig, ax = plt.subplots(figsize=(10, 8))
#         sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
#         st.pyplot(fig)

            
# def local_css(file_name):
#     with open(file_name) as f:
#         st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# local_css("style.css")


# # Run the app
# if __name__ == "__main__":
#     main()
    

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data from CSV
@st.cache_data
def load_data():
    data = pd.read_csv("app_data.csv")
    return data

# Main app function
def main():
    st.title("Cancer Research Data Analysis")
    st.write("This app allows you to explore and visualize cancer research data based on key features.")

    # Load and display data
    data = load_data()
    
    # Selecting only specified columns for analysis
    features = [
        'Appendix_Diameter', 'Management', 'Appendix_on_US_yes', 
        'US_Number', 'Length_of_Stay', 'WBC_Count', 'CRP', 
        'Diagnosis_Presumptive_no appendicitis', 'Alvarado_Score', 
        'Severity_uncomplicated', 'Age', 'Weight', 'Height', 
        'BMI', 'Sex'
    ]
    
    # Filter dataset for the selected features
    data = data[features]
    st.write("### Data Preview (Selected Columns)")
    st.write(data.head())

    # Show basic stats for selected features
    st.write("### Basic Statistics for Selected Features")
    st.write(data.describe())

    # Sidebar filters for specific features
    st.sidebar.header("Filter Options")
    # Example: Filtering based on Age
    if 'Age' in data.columns:
        min_age, max_age = int(data['Age'].min()), int(data['Age'].max())
        age_filter = st.sidebar.slider("Filter by Age", min_value=min_age, max_value=max_age, value=(min_age, max_age))
        data = data[(data['Age'] >= age_filter[0]) & (data['Age'] <= age_filter[1])]
        
    if 'BMI' in data.columns:
        min_bmi, max_bmi = int(data['BMI'].min()), int(data['BMI'].max())
        bmi_filter = st.sidebar.slider("Filter by BMI", min_value=min_bmi, max_value=max_bmi, value=(min_bmi, max_bmi))
        data = data[(data['BMI'] >= bmi_filter[0]) & (data['BMI'] <= bmi_filter[1])]
        
    if 'Height' in data.columns:
        min_height, max_height = int(data['Height'].min()), int(data['Height'].max())
        height_filter = st.sidebar.slider("Filter by height", min_value=min_height, max_value=max_height, value=(min_height,
                                                                                                                 max_height))
        data = data[(data['Height'] >= height_filter[0]) & (data['Height'] <= height_filter[1])]
        
    if 'Weight' in data.columns:
        min_weight, max_weight = int(data['Weight'].min()), int(data['Weight'].max())
        weight_filter = st.sidebar.slider("Filter by weight", min_value=min_weight, max_value=max_weight, value=(min_weight,
                                                                                                                 max_weight))
        data = data[(data['Weight'] >= weight_filter[0]) & (data['Weight'] <= weight_filter[1])]

    # Visualization options
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

    # Count plot for categorical feature (e.g., Management_primary surgical)
    if 'Management_primary surgical' in data.columns:
        st.write("### Count Plot: Management Primary Surgical")
        fig, ax = plt.subplots()
        sns.countplot(data=data, x='Management_primary surgical', ax=ax)
        st.pyplot(fig)

    # Correlation heatmap for numeric features
    st.write("### Correlation Matrix for Numeric Features")
    numeric_features = data.select_dtypes(include=['float64', 'int64']).dropna()
    if numeric_features.empty:
        st.write("No numeric columns available for correlation.")
    else:
        corr = numeric_features.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    main()
