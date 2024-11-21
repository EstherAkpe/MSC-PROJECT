import streamlit as st
import pickle
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

# Load dataset and models
X_train_cvd = pd.read_csv('2train.csv')

def load_pickle(file_name):
    with open(file_name, "rb") as f:
        return pickle.load(f)

cvd_cleaned_data_clean = load_pickle("cvd_cleaned_data_clean.pkl")
best_model = load_pickle("best_model.pkl")
label_encoders = load_pickle("label_encoder.pkl")
scaler = load_pickle("scaler.pkl")
pca = load_pickle("pca.pkl")

age_map = {
    "18-24": 21, "25-29": 27, "30-34": 32, "35-39": 37,
    "40-44": 42, "45-49": 47, "50-54": 52, "55-59": 57,
    "60-64": 62, "65-69": 67, "70-74": 72, "75-79": 77,
    "80+": 85
}
# Add Info Session
st.title("🫀 Heart Disease Risk Predictor")
st.write("""
This application helps you predict the risk of heart disease based on various health and lifestyle factors. 
Use this tool to evaluate potential risks and receive general recommendations for improving heart health.
""")

# About Section
st.markdown("### 📘 About")
st.info("""
This tool uses a machine learning model trained on health and lifestyle data to estimate heart disease risk. 
It evaluates input factors like BMI, alcohol consumption, smoking history, and more to make predictions.
For best results, provide accurate information or use the randomization feature to test various scenarios.
""")

# How to Use Section
st.markdown("### ℹ️ How to Use")
st.markdown("""
1. **Provide Input**: Enter your details in the form below, such as height, weight, and lifestyle habits.
2. **Adjust Factors**: Experiment with different values to understand how they affect the prediction.
3. **Get Prediction**: Click the **Predict** button to see your heart disease risk level.
4. **View Recommendations**: Based on the results, follow suggestions for improving your health.
5. **Randomize or Reset**: Use the buttons in the sidebar to generate sample data or reset to defaults.
""")

# Initialize Session State
if "inputs" not in st.session_state:
    st.session_state.inputs = {
        "height": 170,
        "weight": 70,
        "bmi": 24.22,
        "alcohol_consumption": 2,
        "fruit_consumption": 5,
        "green_vegetables_consumption": 3,
        "fried_potatoes": 1,
        "general_health": 'Good',
        "checkup": 'Within the past year',
        "exercise": "Yes",
        "skin_cancer": "No",
        "other_cancer": "No",
        "depression": "No",
        "diabetes": 'No',
        "arthritis": "No",
        "sex": "Male",
        "smoking_history": "No",
        "age_category": "30-34"
    }

# Function to generate random input values
def generate_random_values():
    random_height = random.randint(100, 500)
    random_weight = random.randint(30, 500)
    random_bmi = round(random_weight / ((random_height / 100) ** 2), 2)
    return {
        "height": random_height,
        "weight": random_weight,
        "bmi": random_bmi,
        "alcohol_consumption": random.randint(0, 100),
        "fruit_consumption": random.randint(0, 100),
        "green_vegetables_consumption": random.randint(0, 100),
        "fried_potatoes": random.randint(0, 100),
        "general_health": random.choice(['Excellent', 'Fair', 'Good', 'Poor', 'Very Good']),
        "checkup": random.choice(['5 or more years ago', 'Never', 'Within the past 2 years',
                                  'Within the past 5 years', 'Within the past year']),
        "exercise": random.choice(["Yes", "No"]),
        "skin_cancer": random.choice(["Yes", "No"]),
        "other_cancer": random.choice(["Yes", "No"]),
        "depression": random.choice(["Yes", "No"]),
        "diabetes": random.choice(['No', 'No, pre-diabetes or borderline diabetes', 
                                    'Yes', 'Yes, but female told only during pregnancy']),
        "arthritis": random.choice(["Yes", "No"]),
        "sex": random.choice(["Male", "Female"]),
        "smoking_history": random.choice(["Yes", 'No']),
        "age_category": random.choice(list(age_map.keys()))
    }

# Sidebar Inputs (reflect session state)
st.sidebar.header("📋 User Input Data")
height = st.sidebar.number_input("Height (cm)", min_value=100, max_value=500, value=st.session_state.inputs["height"], key="height")
weight = st.sidebar.number_input("Weight (kg)", min_value=30, max_value=500, value=st.session_state.inputs["weight"], key="weight")
bmi = round(weight / ((height / 100) ** 2), 2)
# Display BMI and Input Summary
st.sidebar.write(f"💡 **Your BMI:** {bmi}")

alcohol_consumption = st.sidebar.number_input("Alcohol Consumption (drinks/week)", min_value=0, max_value=100, value=st.session_state.inputs["alcohol_consumption"], key="alcohol")
fruit_consumption = st.sidebar.number_input("Fruit Consumption (servings/day)", min_value=0, max_value=100, value=st.session_state.inputs["fruit_consumption"], key="fruit")
green_vegetables_consumption = st.sidebar.number_input("Vegetables Consumption (servings/day)", min_value=0, max_value=100, value=st.session_state.inputs["green_vegetables_consumption"], key="vegetables")
fried_potatoes = st.sidebar.number_input("Fried Potato Consumption (servings/week)", min_value=0, max_value=100, value=st.session_state.inputs["fried_potatoes"], key="potatoes")

general_health = st.sidebar.selectbox("General Health", ['Excellent', 'Fair', 'Good', 'Poor', 'Very Good'], index=['Excellent', 'Fair', 'Good', 'Poor', 'Very Good'].index(st.session_state.inputs["general_health"]))
checkup = st.sidebar.selectbox("Frequency of Checkups", ['5 or more years ago', 'Never', 'Within the past 2 years', 'Within the past 5 years', 'Within the past year'], index=['5 or more years ago', 'Never', 'Within the past 2 years', 'Within the past 5 years', 'Within the past year'].index(st.session_state.inputs["checkup"]))
exercise = st.sidebar.selectbox("Exercise Regularly", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.inputs["exercise"]))
skin_cancer = st.sidebar.selectbox("History of Skin Cancer", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.inputs["skin_cancer"]))
other_cancer = st.sidebar.selectbox("History of Other Cancer", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.inputs["other_cancer"]))
depression = st.sidebar.selectbox("History of Depression", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.inputs["depression"]))
diabetes = st.sidebar.selectbox("History of Diabetes", ['No', 'No, pre-diabetes or borderline diabetes', 'Yes', 'Yes, but female told only during pregnancy'], index=['No', 'No, pre-diabetes or borderline diabetes', 'Yes', 'Yes, but female told only during pregnancy'].index(st.session_state.inputs["diabetes"]))
arthritis = st.sidebar.selectbox("History of Arthritis", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.inputs["arthritis"]))
sex = st.sidebar.selectbox("Sex", ["Male", "Female"], index=["Male", "Female"].index(st.session_state.inputs["sex"]))
smoking_history = st.sidebar.selectbox("Smoking History", ["Yes", "No"], index=["Yes", "No"].index(st.session_state.inputs["smoking_history"]))
age_category = st.sidebar.selectbox("Age Category", list(age_map.keys()), index=list(age_map.keys()).index(st.session_state.inputs["age_category"]))


# Randomize and Reset Buttons
if st.sidebar.button("🎲 Randomize Input"):
    st.session_state.inputs = generate_random_values()

if st.sidebar.button("🔄 Reset Input"):
    st.session_state.inputs = {
        "height": 170,
        "weight": 70,
        "bmi": round(70 / ((170 / 100) ** 2), 2),
        "alcohol_consumption": 2,
        "fruit_consumption": 5,
        "green_vegetables_consumption": 3,
        "fried_potatoes": 1,
        "general_health": 'Good',
        "checkup": 'Within the past year',
        "exercise": "Yes",
        "skin_cancer": "No",
        "other_cancer": "No",
        "depression": "No",
        "diabetes": 'No',
        "arthritis": "No",
        "sex": "Male",
        "smoking_history": "No",
        "age_category": "30-34"
    }

# Predict Button
if st.sidebar.button("🔍 Predict"):
    try:
        # Prepare Input
        new_sample = pd.DataFrame({
            "height_(cm)": [height],
            "weight_(kg)": [weight],
            "bmi": [bmi],
            "alcohol_consumption": [alcohol_consumption],
            "fruit_consumption": [fruit_consumption],
            "green_vegetables_consumption": [green_vegetables_consumption],
            "friedpotato_consumption": [fried_potatoes],
            "general_health": [general_health],
            "checkup": [checkup],
            "exercise": [exercise],
            "skin_cancer": [skin_cancer],
            "other_cancer": [other_cancer],
            "depression": [depression],
            "diabetes": [diabetes],
            "arthritis": [arthritis],
            "sex": [sex],
            "smoking_history": [smoking_history],
            "age_category": [age_category]
        })
        
        # Feature Engineering
        new_sample["obese"] = (new_sample["bmi"] >= 28).astype(int)
        new_sample["healthy_eating_ratio"] = new_sample["fruit_consumption"] / (new_sample["friedpotato_consumption"] + 1)
        new_sample["age_category"] = new_sample["age_category"].map(age_map)

        # Encode Categorical Variables
        for col in ["general_health", "checkup", "exercise", "skin_cancer", "other_cancer", 
                    "depression", "diabetes", "arthritis", "sex", "smoking_history"]:
            new_sample[col] = label_encoders[col].transform(new_sample[col])

        # Scale Numerical Features
        num_cols = ['bmi', 'height_(cm)', 'weight_(kg)', 'fruit_consumption', 
                    'green_vegetables_consumption', 'friedpotato_consumption']
        new_sample[num_cols] = scaler.transform(new_sample[num_cols])

        # Apply PCA
        new_sample_pca = PCA(n_components=8, random_state=42).fit(X_train_cvd).transform(new_sample[X_train_cvd.columns])

        # Prediction
        prediction = best_model.predict(new_sample_pca)
        result = "at risk" if prediction[0] == 1 else "not at risk"
        st.success(f"🫀 **Prediction:** You are {result} of heart disease.")

        # Recommendations
        st.write("### 🏋️ Recommendations")
        if bmi >= 28:
            st.warning("Reduce BMI by engaging in regular physical activity and maintaining a balanced diet.")
        if alcohol_consumption > 7:
            st.warning("Consider reducing alcohol intake to lower cardiovascular risk.")
        if fruit_consumption < 2:
            st.warning("Increase fruit and vegetable consumption for better heart health.")
        if smoking_history == "Yes":
            st.warning("Quit smoking to significantly reduce heart disease risk.")

    except Exception as e:
        st.error(f"❗ Error: {e}")