import streamlit as st
import pandas as pd
from sklearn.neural_network import MLPClassifier
import joblib

# Load dataset
df = pd.read_csv('breast_cancer_data.csv')  # Load the prepared dataset from CSV
X = df.drop(columns=['target'])  # Separate features from the target variable
y = df['target']  # Extract the target variable

# Train model
model = MLPClassifier(hidden_layer_sizes=(50,), activation='relu', solver='adam', max_iter=1000)  # Initialize and configure the MLPClassifier
model.fit(X, y)  # Fit the model to the entire dataset
joblib.dump(model, 'breast_cancer_model.pkl')  # Save the trained model to a file

# Streamlit app
st.markdown(
    """
    <style>
    .title {
        text-align: center;
        font-size: 48px;  /* Increased font size */
        font-weight: bold;
        margin-top: 20px;  /* Reduced margin-top to remove extra space */
        font-family: 'Arial', sans-serif;  /* Changed font to Arial */
    }
    .container {
        display: flex;
        flex-direction: column;
        justify-content: flex-start;
        align-items: center;
        height: 100vh;
        padding: 0;
    }
    .button-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 20px; /* Adjust button gap and positioning */
    }
    </style>
    """, unsafe_allow_html=True)

# Center the title using custom CSS
st.markdown('<div class="title">Breast Cancer Prediction App</div>', unsafe_allow_html=True)

st.write("This app uses a neural network to predict if a tumor is malignant or benign.")  # Add a description to the app

# Initialize session state for storing user input
if 'user_input' not in st.session_state:
    st.session_state.user_input = {feature: 0.0 for feature in X.columns}

# Create five columns for displaying inputs
rows = st.columns(5)  # Five columns for input fields

# User inputs for the selected features
for i, feature in enumerate(X.columns):
    with rows[i % 5]:  # Distribute inputs between five columns
        st.session_state.user_input[feature] = st.number_input(f"{feature}", value=st.session_state.user_input[feature])

# Initialize a variable to hold prediction result
prediction_result = None

# Create buttons and align them
with st.container():
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    
    # Predict button
    if st.button('Predict'):  # When the user clicks the Predict button
        # Check if all the required inputs are valid
        input_data = pd.DataFrame([st.session_state.user_input])  # Convert user inputs to a DataFrame
        model = joblib.load('breast_cancer_model.pkl')  # Load the trained model
        prediction = model.predict(input_data)  # Make a prediction based on user input
        prediction_result = 'Malignant' if prediction[0] == 0 else 'Benign'  # Interpret the prediction result
        
        # Show a prompt message with the prediction result
        st.success(f"Prediction: The tumor is {prediction_result}.")  # Display result as a success message
    
    # Reset button
    if st.button('Reset'):
        # Reset all user inputs to 0 (minimum value)
        for feature in X.columns:
            st.session_state.user_input[feature] = 0.0
        st.experimental_rerun()  # Refresh the app to reflect the reset
    
    st.markdown('</div>', unsafe_allow_html=True)
