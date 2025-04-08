import os
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import joblib

# --- Define File Paths Using os.path.join ---
MODEL_FILE_PATH = os.path.join(os.path.dirname(__file__), 'CombinedModel.sav')
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'label_encoder.sav')
TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), 'Training.csv')
DIABETES_DATA_PATH = os.path.join(os.path.dirname(__file__), 'diabetes.csv')
HEART_DATA_PATH = os.path.join(os.path.dirname(__file__), 'heart.csv')

# --- Load Trained Combined Model ---
try:
    combined_model = joblib.load(MODEL_FILE_PATH)
except FileNotFoundError:
    st.error(f"Error loading the combined model file ('{MODEL_FILE_PATH}'). Please ensure it's in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the combined model: {e}")
    st.stop()

# --- Load Trained LabelEncoder ---
try:
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
except FileNotFoundError:
    st.error(f"Error loading the LabelEncoder file ('{LABEL_ENCODER_PATH}'). Please ensure it's in the correct directory.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading the LabelEncoder: {e}")
    st.stop()

# --- Load Training Data to Get Features ---
try:
    train_data = pd.read_csv(TRAINING_DATA_PATH)
    features_columns = train_data.drop('prognosis', axis=1).columns
except FileNotFoundError:
    st.error(f"Error loading 'Training.csv' ('{TRAINING_DATA_PATH}'). Please ensure it's in the same directory as the app.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading 'Training.csv': {e}")
    st.stop()

# Function to predict diseases based on symptoms
def predict_diseases(symptoms_str):
    symptoms = [s.strip().capitalize() for s in symptoms_str.split(',')]
    input_data = pd.DataFrame(np.zeros((1, len(features_columns)), dtype=int), columns=features_columns)
    for symptom in symptoms:
        symptom_lower_no_space = symptom.lower().replace(' ', '_')
        if symptom_lower_no_space in features_columns:
            input_data[symptom_lower_no_space] = 1

    prediction_encoded = combined_model.predict(input_data)[0]
    predicted_disease = label_encoder.inverse_transform([prediction_encoded])[0]
    return predicted_disease

# Function to predict diabetes based on symptoms
def predict_diabetes(symptoms, features, svm_classifier, scaler):
    input_data = [0] * len(features.columns)
    for symptom in symptoms:
        if symptom in features.columns:
            index = features.columns.get_loc(symptom)
            input_data[index] = 1
    input_data = pd.DataFrame([input_data], columns=features.columns)
    standardized_data = scaler.transform(input_data)
    predictions = svm_classifier.predict(standardized_data)
    return predictions

# Sidebar for navigation
with st.sidebar:
    st.title("PolyDisease Predictor")
    selected = st.radio("Select Predictor", ["🩸 Diabetes Prediction", "❤️ Heart Disease Prediction", "🦠 Multiple Disease Prediction"])

# Load data
if selected == "🩸 Diabetes Prediction" or selected == "❤️ Heart Disease Prediction":
    try:
        diabetes_data = pd.read_csv(DIABETES_DATA_PATH)
        heart_disease_data = pd.read_csv(HEART_DATA_PATH)

        # Training SVM model for diabetes prediction
        X_diabetes = diabetes_data.drop(columns='Outcome', axis=1)
        Y_diabetes = diabetes_data['Outcome']
        scaler_diabetes = StandardScaler()
        scaler_diabetes.fit(X_diabetes)
        X_diabetes_standardized = scaler_diabetes.transform(X_diabetes)
        X_diabetes_train, X_diabetes_test, Y_diabetes_train, Y_diabetes_test = train_test_split(X_diabetes_standardized, Y_diabetes, test_size=0.2, stratify=Y_diabetes, random_state=2)
        svm_classifier_diabetes = SVC(kernel='linear')
        svm_classifier_diabetes.fit(X_diabetes_train, Y_diabetes_train)

        # Training RandomForest model for heart disease prediction
        X_heart = heart_disease_data.drop(columns='target', axis=1)
        Y_heart = heart_disease_data['target']
        X_heart_train, X_heart_test, Y_heart_train, Y_heart_test = train_test_split(X_heart, Y_heart, test_size=0.2, stratify=Y_heart, random_state=8)
        rf_classifier_heart = RandomForestClassifier(n_estimators=100)
        rf_classifier_heart.fit(X_heart_train, Y_heart_train)

    except FileNotFoundError as e:
        st.error(f"Error loading data for Diabetes or Heart Disease Prediction: {e.filename} not found. Please ensure the files are in the same directory as the app.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading data for Diabetes or Heart Disease Prediction: {e}")
        st.stop()

# Multiple Disease Prediction Page
if selected == "🦠 Multiple Disease Prediction":
    st.title("Multiple Disease Prediction using Symptoms")
    symptoms_input = st.text_area("Enter your symptoms (comma-separated):", placeholder="e.g., Itching, Skin Rash, Nodal Skin Eruptions")
    predict_button = st.button("Predict Disease")

    if predict_button and symptoms_input:
        if symptoms_input.strip():
            try:
                predicted_disease = predict_diseases(symptoms_input)
                st.success(f"Predicted Disease: {predicted_disease}")
            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please enter some symptoms.")

# Diabetes Prediction Page
if selected == "🩸 Diabetes Prediction":
    st.title("Diabetes Prediction using SVM")
    pregnancies = st.number_input("Number of Pregnancies")
    glucose = st.number_input("Glucose Level")
    blood_pressure = st.number_input("Blood Pressure value")
    skin_thickness = st.number_input("Skin Thickness value")
    insulin = st.number_input("Insulin Level")
    bmi = st.number_input("BMI value")
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function value")
    age = st.number_input("Age of the Person")

    if st.button("Diabetes Test Result"):
        diab_prediction = predict_diabetes([pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age], X_diabetes, svm_classifier_diabetes, scaler_diabetes)
        diab_diagnosis = "The person is diabetic" if diab_prediction[0] == 1 else "The person is not diabetic"
        st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == "❤️ Heart Disease Prediction":
    st.title("Heart Disease Prediction using RandomForest")
    age_heart = st.number_input("Age")
    sex_heart = st.number_input("Sex")
    cp_heart = st.number_input("Chest Pain types")
    trestbps_heart = st.number_input("Resting Blood Pressure")
    chol_heart = st.number_input("Serum Cholestoral in mg/dl")
    fbs_heart = st.number_input("Fasting Blood Sugar > 120 mg/dl")
    restecg_heart = st.number_input("Resting Electrocardiographic results")
    thalach_heart = st.number_input("Maximum Heart Rate achieved")
    exang_heart = st.number_input("Exercise Induced Angina")
    oldpeak_heart = st.number_input("ST depression induced by exercise")
    slope_heart = st.number_input("Slope of the peak exercise ST segment")
    ca_heart = st.number_input("Major vessels colored by flourosopy")
    thal_heart = st.number_input("thal: 0 = normal; 1 = fixed defect; 2 = reversable defect")

    if st.button("Heart Disease Test Result"):
        heart_prediction = rf_classifier_heart.predict(
            [[age_heart, sex_heart, cp_heart, trestbps_heart, chol_heart, fbs_heart, restecg_heart, thalach_heart, exang_heart, oldpeak_heart, slope_heart, ca_heart, thal_heart]]
        )
        heart_diagnosis = "The person is having heart disease" if heart_prediction[0] == 1 else "The person does not have any heart disease"
        st.success(heart_diagnosis)