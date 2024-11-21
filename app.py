import streamlit as st
import pickle
import pandas as pd

# Load the saved model, scaler, PCA, and any other preprocessors
with open("best_model.pkl", "rb") as f:
    best_model = pickle.load(f)
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)
with open("pca.pkl", "rb") as f:
    pca = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Define the Streamlit app
st.title("Heart Disease Prediction with PCA")
st.write("Enter your details below to predict the likelihood of heart disease.")

# Input fields for user data
height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=170.0, step=0.1)
weight = st.number_input("Weight (kg)", min_value=20.0, max_value=200.0, value=70.0, step=0.1)
bmi = weight / ((height / 100) ** 2)
st.write(f"Calculated BMI: {bmi:.2f}")

alcohol_consumption = st.number_input("Alcohol Consumption (drinks per week)", min_value=0, max_value=50, value=1)
fruit_consumption = st.number_input("Fruit Consumption (servings per week)", min_value=0, max_value=50, value=5)
vegetables_consumption = st.number_input("Vegetables Consumption (servings per week)", min_value=0, max_value=50, value=5)
friedpotato_consumption = st.number_input("Fried Potato Consumption (servings per week)", min_value=0, max_value=50, value=1)

# Categorical inputs
general_health = st.selectbox("General Health", ["Excellent", "Very good", "Good", "Fair", "Poor"])
exercise = st.radio("Exercise Regularly?", ["Yes", "No"])
smoking_history = st.selectbox("Smoking History", ["Never", "Former", "Current"])

# Encode categorical features
encoded_health = label_encoder.transform([general_health])[0]
encoded_exercise = 1 if exercise == "Yes" else 0
encoded_smoking = label_encoder.transform([smoking_history])[0]

# Prepare input data
input_data = pd.DataFrame(
    {
        "height_(cm)": [height],
        "weight_(kg)": [weight],
        "bmi": [bmi],
        "alcohol_consumption": [alcohol_consumption],
        "fruit_consumption": [fruit_consumption],
        "green_vegetables_consumption": [vegetables_consumption],
        "friedpotato_consumption": [friedpotato_consumption],
        "general_health": [encoded_health],
        "exercise": [encoded_exercise],
        "smoking_history": [encoded_smoking],
    }
)

# Standardize numerical features
numerical_features = ["height_(cm)", "weight_(kg)", "bmi", "fruit_consumption", "green_vegetables_consumption", "friedpotato_consumption"]
input_data[numerical_features] = scaler.transform(input_data[numerical_features])

# Apply PCA
input_data_pca = pca.transform(input_data)

# Predict using the trained model
if st.button("Predict"):
    prediction = best_model.predict(input_data_pca)[0]
    prediction_proba = best_model.predict_proba(input_data_pca)[0][1]

    if prediction == 1:
        st.error(f"High likelihood of heart disease (Probability: {prediction_proba:.2f})")
    else:
        st.success(f"Low likelihood of heart disease (Probability: {1 - prediction_proba:.2f})")
