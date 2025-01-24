import pandas as pd
import numpy as np
import streamlit as st
import pickle

# Load the model
pickle_in = open('model.pkl', 'rb')
model = pickle.load(pickle_in)

# Prediction function
def prediction(data):
    prediction = model.predict(data)
    return prediction

# User input function
def user_input_features():
    # Set the app title and add a background image
    st.markdown(
        """
        <style>
        .reportview-container {
            background: url("Background.jpg.jpg"); 
            background-size: cover;
        }
        .sidebar .sidebar-content {
            background: #F5F5F5;
        }
        .stButton>button {
            background-color: #FF4B4B;
            color: white;
            border-radius: 5px;
        }
        .stTitle {
            color: #2C3E50;
            text-align: center;
            font-family: 'Arial';
            font-weight: bold;
        }
        .stHeader {
            text-align: center;
            color: #4B6584;
        }
        </style>
        """, unsafe_allow_html=True
    )

    # Title and brief description
    st.title('Stroke Prediction Web App')
    st.write("""
    According to the World Health Organization (WHO), stroke is the 2nd leading cause of death globally, 
    responsible for approximately 11% of total deaths. This WebApp is used to predict whether a patient is 
    likely to have a stroke based on their input data.
    """)
    
    # Add an image related to stroke (replace URL with your image)
    st.image('stroke.jpg', use_column_width=True)

    st.sidebar.header('Input Patient Data')
    
    # Sidebar inputs for user data
    gender = st.sidebar.selectbox('Gender', ('Male', 'Female', 'Other'))
    hypertension = st.sidebar.selectbox('Hypertension', ('No', 'Yes'))
    heart_disease = st.sidebar.selectbox('Heart Disease', ('No', 'Yes'))
    ever_married = st.sidebar.selectbox('Married', ('Yes', 'No'))
    work_type = st.sidebar.selectbox('Work Type', ('Private', 'Self-employed', 'Govt_job', 'children', 'Never_worked'))
    residence_type = st.sidebar.selectbox('Residence Type', ('Urban', 'Rural'))
    smoking_status = st.sidebar.selectbox('Smoking Status', ('never smoked', 'Unknown', 'formerly smoked', 'smokes'))
    age = st.sidebar.slider('Age', min_value=1, max_value=82)
    avg_glucose_level = st.sidebar.slider('Average Glucose Level', min_value=55.12, max_value=272.00)
    bmi = st.sidebar.slider('BMI', min_value=10.00, max_value=98.00)
    
    # Creating a DataFrame from the inputs
    data = {'gender': gender,
            'hypertension': hypertension,
            'heart_disease': heart_disease,
            'ever_married': ever_married,
            'work_type': work_type,
            'Residence_type': residence_type,
            'smoking_status': smoking_status,
            'age': age,
            'avg_glucose_level': avg_glucose_level,
            'bmi': bmi}
    
    features = pd.DataFrame(data, index=[0])
    return features

# Get user input
input_df = user_input_features()

# Load the dataset
raw_data = pd.read_csv('clean_data.csv')
data = raw_data.drop(columns=['stroke'])

# Concatenate user input with the dataset
df = pd.concat([input_df, data], axis=0)

# Encoding categorical features
encode = ['gender', 'hypertension', 'heart_disease', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']
for col in encode:
    dummy = pd.get_dummies(df[col], prefix=col)
    df = pd.concat([df, dummy], axis=1)
    del df[col]

df = df[:1]  # Keep only the first row (user input)

# Display the input features if "Predict" button is pressed
if st.button("Predict Stroke"):
    st.subheader('Patient Data')
    st.write(df)

    st.subheader('Prediction Result')
    result = prediction(df)
    st.success(f"The model predicts: {'Stroke' if result[0] == 1 else 'No Stroke'}")

# Display another image below the prediction (optional)
#st.image('https://www.stroke.org/images/stroke-prevention-tips.jpg', use_column_width=True)
