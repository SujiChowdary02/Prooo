import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from ydata_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import base64
import os

# Load dataset or create an empty DataFrame
if os.path.exists('./customerData.csv'):
    df = pd.read_csv('customerData.csv')
else:
    df = pd.DataFrame()

# Sidebar navigation with different options
with st.sidebar:
    st.image("https://www.onepointltd.com/wp-content/uploads/2020/03/inno2.png")
    st.title("Customer Product Prediction")
    choice = st.radio("Navigation", ["Upload", "Profiling", "Modelling", "Download"])
    st.info("Predict the number of products customers will buy in a given time range.")

# Upload section to allow users to upload a dataset
if choice == "Upload":
    st.title("Upload Your Dataset")
    file = st.file_uploader("Upload Your Dataset", type=["csv"])
    if file:
        df = pd.read_csv(file)
        st.dataframe(df)
        st.success('File uploaded successfully!')

# Profiling section for exploratory data analysis
if choice == "Profiling":
    st.title("Exploratory Data Analysis")
    if not df.empty:
        profile_df = ProfileReport(df)
        st_profile_report(profile_df)

        # Add a button to download the profiling report
        download_button = st.button("Download Profile Report")
        if download_button:
            # Get the HTML report
            profile_html = profile_df.to_html()
            
            # Convert HTML string to bytes
            profile_bytes = profile_html.encode('utf-8')

            # Encode the bytes to base64
            profile_as_base64 = base64.b64encode(profile_bytes).decode('utf-8')

            # Create a download link for the base64-encoded HTML
            href = f'<a href="data:text/html;base64,{profile_as_base64}" download="profile.html">Download Profile</a>'
            st.markdown(href, unsafe_allow_html=True)
# Modelling section for model training and prediction
if choice == "Modelling":
    if not df.empty:
        st.title("Modeling")

        # Select target column for classification
        target_column = st.selectbox('Select Target Column', df.columns)

        # Choose time range for prediction
        time_range = st.slider('Select Time Range (1 to 6 months)', 1, 6)

        # Add a button to generate the model
        if st.button('Generate Model'):
            # Preprocessing: Splitting data into features and target variable
            X = df.drop(columns=[target_column])
            y = df[target_column]

            # Preprocessing steps for classification
            categorical_features = ['gender', 'payment_method', 'shopping_mall']
            numeric_features = ['age', 'quantity', 'price']

            categorical_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                ('onehot', OneHotEncoder(handle_unknown='ignore'))
            ])

            numeric_transformer = Pipeline(steps=[
                ('imputer', SimpleImputer(strategy='median')),
                ('scaler', StandardScaler())
            ])

            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, categorical_features),
                    ('num', numeric_transformer, numeric_features)
                ]
            )

            # Model building
            model = RandomForestClassifier(n_estimators=100, random_state=42)

            # Full pipeline with preprocessing and model
            pipeline = Pipeline(steps=[
                ('preprocessor', preprocessor),
                ('model', model)
            ])

            # Train the model
            pipeline.fit(X, y)

            # Make predictions for the selected time range
            st.write(f"Predicted classes for customers in {time_range} months:")
            # Create a sample input for prediction (you can adjust this based on your use case)
            sample_input = pd.DataFrame({
                'customer_id': [1],
                'gender': ['Male'],
                'age': [30],
                'quantity': [5],
                'price': [50],
                'payment_method': ['Credit Card'],
                'shopping_mall': ['Mall A']
            })
            prediction = pipeline.predict(sample_input)
            st.write(prediction)

    else:
        st.warning("Please upload a dataset for modeling.")

# Download section
if choice == "Download":
    if os.path.exists('best_model.pkl'):
        st.download_button('Download Model', 'best_model.pkl', label="Download Model File")
    else:
        st.warning("No model available to download. Please train a model first.")
