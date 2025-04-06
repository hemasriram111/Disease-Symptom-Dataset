import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the pre-trained model and encoder with error handling
try:
    with open('disease_predictor_rf.pkl', 'rb') as file:
        rf_model = pickle.load(file)
    st.write("Model loaded successfully. Type:", type(rf_model))
except Exception as e:
    st.error(f"Error loading model: {e}")
    st.stop()

try:
    with open('label_encoder.pkl', 'rb') as file:
        label_encoder = pickle.load(file)
    st.write("Label encoder loaded successfully.")
except Exception as e:
    st.error(f"Error loading label encoder: {e}")
    st.stop()

# Load the symptom columns from your training data
X_train = pd.read_csv('X_train.csv')
symptom_columns = X_train.columns.tolist()

# Streamlit UI
st.title("Disease Prediction App")

st.write("Select more than 4 symptoms to predict the disease:")

# Multi-select dropdown for symptoms
selected_symptoms = st.multiselect(
    "Symptoms",
    options=symptom_columns,
    default=symptom_columns[:5]  # Default to first 5 symptoms
)

if st.button("Predict"):
    if len(selected_symptoms) <= 4:
        st.error("Please select more than 4 symptoms.")
    else:
        # Create input vector (assuming binary encoding: 1 if symptom selected, 0 otherwise)
        input_vector = np.zeros(len(symptom_columns))
        for symptom in selected_symptoms:
            if symptom in symptom_columns:
                input_vector[symptom_columns.index(symptom)] = 1

        # Predict disease
        try:
            prediction = rf_model.predict([input_vector])[0]
            probability = rf_model.predict_proba([input_vector]).max()

            # Decode the prediction
            predicted_disease = label_encoder.inverse_transform([prediction])[0]

            # Display results
            st.success(f"Predicted Disease: **{predicted_disease}**")
            st.write(f"Confidence Score: **{probability:.2f}**")
        except Exception as e:
            st.error(f"Prediction error: {e}")

# Add some information
st.write("### About")
st.write("This app uses a Random Forest model trained on symptom data to predict diseases. Select symptoms from the dropdown and click 'Predict' to get results.")