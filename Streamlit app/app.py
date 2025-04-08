# -*- coding: utf-8 -*-
"""
Created on Mon Nov  6 15:22:52 2023

@author: sudha
"""

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
import pickle  # Import the pickle library

# --- Define File Paths Using os.path.join ---
MODEL_FILE_PATH = os.path.join(os.path.dirname(__file__), 'CombinedModel.sav')
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'label_encoder.sav')
TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), 'Training.csv')
DIABETES_MODEL_PATH = os.path.join(os.path.dirname(__file__), './DiabetesModel.sav')
HEART_DISEASE_MODEL_PATH = os.path.join(os.path.dirname(__file__), './HeartModel.sav')

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

# --- Load Training Data to Get Features for Multiple Disease Prediction ---
try:
    train_data = pd.read_csv(TRAINING_DATA_PATH)
    features_columns = train_data.drop('prognosis', axis=1).columns
except FileNotFoundError:
    st.error(f"Error loading 'Training.csv' ('{TRAINING_DATA_PATH}'). Please ensure it's in the same directory as the app.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading 'Training.csv': {e}")
    st.stop()

# Function to predict multiple diseases based on symptoms
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

# Sidebar for navigation
with st.sidebar:
    st.title("PolyDisease Predictor")
    selected = st.radio("Select Predictor", ["🦠 Multiple Disease Prediction", "🩸 Diabetes Prediction", "❤️ Heart Disease Prediction"])

# Multiple Disease Prediction Page
if selected == "🦠 Multiple Disease Prediction":
    st.title("Multiple Disease Prediction using Symptoms")
    symptoms_input = st.text_area("Enter your symptoms (comma-separated):", placeholder="e.g. Itching, Skin Rash, Nodal Skin Eruptions")
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
    st.title("Diabetes Prediction using ML")

    try:
        diabetes_model = pickle.load(open(DIABETES_MODEL_PATH, 'rb'))

        pregnancies = st.number_input("Number of Pregnancies", min_value=0)
        glucose = st.number_input("Glucose Level", min_value=0)
        blood_pressure = st.number_input("Blood Pressure value", min_value=0)
        skin_thickness = st.number_input("Skin Thickness value", min_value=0)
        insulin = st.number_input("Insulin Level", min_value=0)
        bmi = st.number_input("BMI value", min_value=0.0)
        diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function value", min_value=0.0)
        age = st.number_input("Age of the Person", min_value=0)

        if st.button("Diabetes Test Result"):
            diab_prediction = diabetes_model.predict(
                [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
            )
            diab_diagnosis = "The person is diabetic" if diab_prediction[0] == 1 else "The person is not diabetic"
            st.success(diab_diagnosis)

    except FileNotFoundError:
        st.error(f"Error loading the diabetes model file ('{DIABETES_MODEL_PATH}'). Please ensure it's in the 'models' directory.")
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the diabetes model: {e}")

# Heart Disease Prediction Page
if selected == "❤️ Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")

    try:
        heart_disease_model = pickle.load(open(HEART_DISEASE_MODEL_PATH, 'rb'))

        age_heart = st.number_input("Age", min_value=0)
        sex_heart = st.number_input("Sex", min_value=0, max_value=1)
        cp_heart = st.number_input("Chest Pain types", min_value=0, max_value=3)
        trestbps_heart = st.number_input("Resting Blood Pressure", min_value=0)
        chol_heart = st.number_input("Serum Cholestoral in mg/dl", min_value=0)
        fbs_heart = st.number_input("Fasting Blood Sugar > 120 mg/dl", min_value=0, max_value=1)
        restecg_heart = st.number_input("Resting Electrocardiographic results", min_value=0, max_value=2)
        thalach_heart = st.number_input("Maximum Heart Rate achieved", min_value=0)
        exang_heart = st.number_input("Exercise Induced Angina", min_value=0, max_value=1)
        oldpeak_heart = st.number_input("ST depression induced by exercise", min_value=0.0)
        slope_heart = st.number_input("Slope of the peak exercise ST segment", min_value=0, max_value=2)
        ca_heart = st.number_input("Major vessels colored by flourosopy", min_value=0, max_value=4)
        thal_heart = st.number_input("thal: 0 = normal; 1 = fixed defect; 2 = reversable defect", min_value=0, max_value=3)

        if st.button("Heart Disease Test Result"):
            heart_prediction = heart_disease_model.predict(
                [[age_heart, sex_heart, cp_heart, trestbps_heart, chol_heart, fbs_heart, restecg_heart, thalach_heart, exang_heart, oldpeak_heart, slope_heart, ca_heart, thal_heart]]
            )
            heart_diagnosis = "The person is having heart disease" if heart_prediction[0] == 1 else "The person does not have any heart disease"
            st.success(heart_diagnosis)

    except FileNotFoundError:
        st.error(f"Error loading the heart disease model file ('{HEART_DISEASE_MODEL_PATH}'). Please ensure it's in the 'models' directory.")
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the heart disease model: {e}")