import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import mysql.connector
from mysql.connector import errorcode
from datetime import datetime
from utils.ui import set_bg_from_url
from utils.db import create_db_connection
from utils.model_loading import load_combined_model, load_label_encoder, load_training_data_columns, load_diabetes_model, load_heart_disease_model
from utils.prediction import predict_diseases, add_new_symptom_column
from utils.feedback import insert_feedback_multiple, insert_feedback_diabetes, insert_feedback_heart

st.set_page_config(page_title="Multiple Disease Prediction App")
set_bg_from_url(
    "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQB5z1mX8yxvSR0LwZscejLXVDU-_nCS5AYCA&s"
)

train_df, features_columns = load_training_data_columns()
diabetes_model = load_diabetes_model()
heart_model = load_heart_disease_model()
combined_model = load_combined_model()
label_encoder = load_label_encoder()

if "feedback_data" not in st.session_state:
    st.session_state["feedback_data"] = []

with st.sidebar:
    st.title("PolyDisease Predictor")
    selected = st.radio(
        "Select Predictor",
        ["🦠 Multiple Disease Prediction", "🩸 Diabetes Prediction", "❤️ Heart Disease Prediction"],
    )

if selected == "🦠 Multiple Disease Prediction":
    st.title("Multiple Disease Prediction using Symptoms")

    symptoms_input = st.text_area("Enter your symptoms (comma-separated):", placeholder="e.g. Itching, Skin Rash, Nodal Skin Eruptions")
    predict_button = st.button("Predict Disease")

    mydb = create_db_connection()
    if mydb and mydb.is_connected():
        st.success("Database connection successful!")
    else:
        st.error("Database connection failed.")
        st.stop()

    if predict_button and symptoms_input:
        if symptoms_input.strip():
            try:
                predicted_disease = predict_diseases(symptoms_input, features_columns, combined_model, label_encoder)
                st.success(f"Predicted Disease: {predicted_disease}")

                st.session_state["predicted_disease"] = predicted_disease
                st.session_state["symptoms_list"] = [s.strip().lower().replace(' ', '_') for s in symptoms_input.split(',')]

                if mydb and mydb.is_connected():
                    mycursor = mydb.cursor()
                    mycursor.execute("DESCRIBE feedback_multiple;")
                    existing_columns_info = mycursor.fetchall()
                    existing_db_columns = [col[0] for col in existing_columns_info]
                    mycursor.close()

                    for symptom in st.session_state["symptoms_list"]:
                        safe_symptom = ''.join(c if c.isalnum() or c == '_' else '_' for c in symptom)
                        if safe_symptom not in existing_db_columns and safe_symptom not in ['id', 'prognosis', 'user_feedback', 'feedback_timestamp', 'correct_diagnosis']:
                            if add_new_symptom_column(mydb, symptom):
                                train_df, features_columns = load_training_data_columns()
                                print(f"Updated features_columns after adding '{symptom}': {features_columns}")
                            else:
                                st.error(f"Failed to add symptom column: {symptom}. The feedback might not be saved correctly.")
                                st.stop()

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
                st.stop()
        else:
            st.warning("Please enter some symptoms.")
            st.stop()

    if st.session_state.get("predicted_disease"):
        col1, col2 = st.columns(2)

        with col1:
            if st.button("👍 Correct", key="correct_multiple_btn"):
                insert_feedback_multiple(mydb, st.session_state.get("symptoms_list"), st.session_state.get("predicted_disease"), True, None )

        with col2:
            incorrect_feedback_clicked = st.button("👎 Incorrect", key="incorrect_multiple_btn")
            if incorrect_feedback_clicked:
                st.session_state["incorrect_feedback"] = True

        if st.session_state.get("incorrect_feedback"):
            correct_disease_input = st.text_input("Correct Disease (optional):", "", key="correct_disease_multiple_input")
            if st.button("Submit Correct Disease", key="submit_correct_multiple_btn"):
                symptoms_list = st.session_state.get("symptoms_list")
                predicted_disease = st.session_state.get("predicted_disease")
                insert_feedback_multiple(mydb, symptoms_list, predicted_disease, False, correct_disease_input)
                st.info("Thank you for the corrected information!")
                st.session_state["incorrect_feedback"] = False
                        
if "diab_diagnosis" not in st.session_state:
    st.session_state.diab_diagnosis = None

if selected == "🩸 Diabetes Prediction":
    st.title("Diabetes Prediction using ML")
    mydb = create_db_connection()

    if not mydb or not mydb.is_connected():
        st.error("Failed to connect to the database for feedback.")

    try:
        diabetes_model = load_diabetes_model()

        col1, col2, col3 = st.columns(3)
        with col1:
            pregnancies = st.number_input("Pregnancies", min_value=0)
            glucose = st.number_input("Glucose", min_value=0)
            blood_pressure = st.number_input("Blood Pressure", min_value=0)
        with col2:
            skin_thickness = st.number_input("Skin Thickness", min_value=0)
            insulin = st.number_input("Insulin", min_value=0)
            bmi = st.number_input("BMI", min_value=0.0)
        with col3:
            diabetes_pedigree_function = st.number_input("Pedigree Function", min_value=0.0)
            age = st.number_input("Age", min_value=0)
            st.empty() 

        if st.button("Diabetes Test Result", key="diabetes_test_result_button"):
            input_features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
            diab_prediction = diabetes_model.predict([input_features])
            diab_diagnosis = diab_prediction[0]
            st.session_state.diab_diagnosis = diab_diagnosis

            if diab_diagnosis == 1:
                st.success("Prediction Result: Diabetic")
            else:
                st.success("Prediction Result: Not Diabetic")

        if st.session_state.get('diab_diagnosis') is not None:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Correct", key=f"correct_diabetes_button_{st.session_state.get('diab_diagnosis')}"):
                    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
                    predicted = st.session_state.diab_diagnosis
                    feedback_value = True
                    if insert_feedback_diabetes(create_db_connection(), "feedback_diabetes", input_data, predicted, feedback_value):
                        st.success("Thank you for your feedback!")
                    else:
                        st.error("Error submitting feedback.")

            with col2:
                if st.button("👎 Incorrect", key=f"incorrect_diabetes_button_{st.session_state.get('diab_diagnosis')}"):
                    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
                    predicted = st.session_state.diab_diagnosis
                    feedback_value = False
                    correct_diagnosis_input = st.text_input("Correct Diagnosis (optional):", "", key=f"correct_diagnosis_diabetes_{st.session_state.get('diab_diagnosis')}")
                    if st.button("Submit Correct Diagnosis", key=f"submit_correct_diabetes_button_{st.session_state.get('diab_diagnosis')}"):
                        if insert_feedback_diabetes(create_db_connection(), "feedback_diabetes", input_data, predicted, feedback_value):
                            st.info("Thank you for the corrected information!")
                        else:
                            st.error("Error submitting feedback.")

    except FileNotFoundError:
        st.error(f"Error loading the diabetes model file ('DIABETES_MODEL_PATH'). Please ensure it's in the 'models' directory.")
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the diabetes model: {e}")

    if mydb is None:
        st.error("Failed to connect to the database for feedback.")

if selected == "❤️ Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")

    mydb = create_db_connection()

    try:
        heart_disease_model = load_heart_disease_model()

        col1, col2, col3 = st.columns(3)
        with col1:
            age_heart = st.number_input("Age", min_value=0)
            sex_heart = st.number_input("Sex", min_value=0, max_value=1)
            cp_heart = st.number_input("Chest Pain", min_value=0, max_value=3)
            trestbps_heart = st.number_input("Resting BP", min_value=0)
            chol_heart = st.number_input("Cholesterol", min_value=0)
        with col2:
            fbs_heart = st.number_input("Fasting BS (>120)", min_value=0, max_value=1)
            restecg_heart = st.number_input("Resting ECG", min_value=0, max_value=2)
            thalach_heart = st.number_input("Max Heart Rate", min_value=0)
            exang_heart = st.number_input("Exercise Angina", min_value=0, max_value=1)
        with col3:
            oldpeak_heart = st.number_input("Oldpeak", min_value=0.0)
            slope_heart = st.number_input("Slope", min_value=0, max_value=2)
            ca_heart = st.number_input("Major Vessels", min_value=0, max_value=4)
            thal_heart = st.number_input("Thalassemia", min_value=0, max_value=3)
            st.empty()

        if st.button("Heart Disease Test Result"):
            input_features = [age_heart, sex_heart, cp_heart, trestbps_heart, chol_heart, fbs_heart, restecg_heart, thalach_heart, exang_heart, oldpeak_heart, slope_heart, ca_heart, thal_heart]
            heart_prediction = heart_disease_model.predict([input_features])
            st.session_state.heart_diagnosis = int(heart_prediction[0])

            if heart_prediction[0] == 1:
                st.success("Prediction Result: Has Heart Disease")
            else:
                st.success("Prediction Result: No Heart Disease")

        if st.session_state.get('heart_diagnosis') is not None:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Correct", key=f"correct_heart_button_{st.session_state.get('heart_diagnosis')}"):
                    input_data = [age_heart, sex_heart, cp_heart, trestbps_heart, chol_heart, fbs_heart, restecg_heart, thalach_heart, exang_heart, oldpeak_heart, slope_heart, ca_heart, thal_heart]
                    predicted = st.session_state.heart_diagnosis
                    feedback_value = True
                    if insert_feedback_heart(mydb, "feedback_heart", input_data, predicted, feedback_value):
                        st.success("Thank you for your feedback!")
                    else:
                        st.error("Error submitting feedback.")

            with col2:
                if st.button("👎 Incorrect", key=f"incorrect_heart_button_{st.session_state.get('heart_diagnosis')}"):
                    input_data = [age_heart, sex_heart, cp_heart, trestbps_heart, chol_heart, fbs_heart, restecg_heart, thalach_heart, exang_heart, oldpeak_heart, slope_heart, ca_heart, thal_heart]
                    predicted = st.session_state.heart_diagnosis
                    feedback_value = False
                    correct_diagnosis_input = st.text_input("Correct Diagnosis (optional):", "", key=f"correct_diagnosis_heart_{st.session_state.get('heart_diagnosis')}")
                    if st.button("Submit Correct Diagnosis", key=f"submit_correct_heart_button_{st.session_state.get('heart_diagnosis')}"):
                        if insert_feedback_heart(mydb, "feedback_heart", input_data, predicted, feedback_value):
                            st.info("Thank you for the corrected information!")
                        else:
                            st.error("Error submitting feedback.")

    except FileNotFoundError:
        st.error(f"Error loading the heart disease model file ('HEART_DISEASE_MODEL_PATH'). Please ensure it's in the 'models' directory.")
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the heart disease model: {e}")

    if mydb is None:
        st.error("Failed to connect to the database for feedback.")
