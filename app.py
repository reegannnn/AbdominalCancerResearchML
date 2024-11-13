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
    st.write("This app allows you to explore and visualize cancer research data.")

    # Load and display data
    data = load_data()
    st.write("### Data Preview")
    st.write(data.head())

    # Show basic stats
    st.write("### Basic Data Statistics")
    st.write(data.describe())

    # Sidebar filters (example for numerical filtering)
    st.sidebar.header("Filter Options")
    if 'Age' in data.columns:
        min_age, max_age = int(data['Age'].min()), int(data['Age'].max())
        age_filter = st.sidebar.slider("Filter by Age", min_value=min_age, max_value=max_age, value=(min_age, max_age))
        data = data[(data['Age'] >= age_filter[0]) & (data['Age'] <= age_filter[1])]

    # Visualization options
    st.write("## Data Visualizations")
    
    # Count plot for a categorical column (example for 'Diagnosis')
    if 'Diagnosis' in data.columns:
        st.write("### Diagnosis Distribution")
        fig, ax = plt.subplots()
        sns.countplot(data=data, x='Diagnosis', ax=ax)
        st.pyplot(fig)

    # Scatter plot example for numerical columns (example: 'Age' vs 'Tumor Size')
    if 'Age' in data.columns and 'Tumor_Size' in data.columns:
        st.write("### Scatter Plot: Age vs Tumor Size")
        fig, ax = plt.subplots()
        sns.scatterplot(data=data, x='Age', y='Tumor_Size', ax=ax)
        st.pyplot(fig)

    # Correlation heatmap
    st.write("### Correlation Matrix")
    corr = data.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(corr, annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Run the app
if __name__ == "__main__":
    main()