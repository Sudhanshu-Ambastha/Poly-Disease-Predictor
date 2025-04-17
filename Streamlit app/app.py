import os
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import pickle
import mysql.connector
from mysql.connector import errorcode
from datetime import datetime

def set_bg_from_url(url, opacity=1):
    st.markdown(
        f"""
        <style>
            body {{
                background: url('{url}') no-repeat center center fixed;
                background-size: cover;
                opacity: 0.875;
            }}
        </style>
        """,
        unsafe_allow_html=True
    )


# --- MUST BE THE FIRST STREAMLIT COMMAND ---
st.set_page_config(page_title="Multiple Disease Prediction App")

# Set background image from URL
set_bg_from_url("https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQB5z1mX8yxvSR0LwZscejLXVDU-_nCS5AYCA&s")



# --- Define File Paths ---
MODEL_FILE_PATH = os.path.join(os.path.dirname(__file__), 'CombinedModel.sav')
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), 'label_encoder.sav')
TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), 'Training.csv')
DIABETES_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'DiabetesModel.sav')
HEART_DISEASE_MODEL_PATH = os.path.join(os.path.dirname(__file__), 'HeartModel.sav')

@st.cache_resource
def create_db_connection():
    try:
        connection = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            # database=st.secrets["mysql"].get("database", None)
        )
        if connection.is_connected():
            cursor = connection.cursor()
            # Create the database if it doesn't exist
            cursor.execute("CREATE DATABASE IF NOT EXISTS poly_disease_predictor")
            cursor.execute("USE poly_disease_predictor")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_diabetes (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    pregnancies INT,
                    glucose INT,
                    blood_pressure INT,
                    skin_thickness INT,
                    insulin INT,
                    bmi FLOAT,
                    diabetes_pedigree_function FLOAT,
                    age INT,
                    predicted_outcome VARCHAR(255) NOT NULL,
                    user_feedback BOOLEAN,
                    feedback_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_heart (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    age INT,
                    sex INT,
                    cp INT,
                    trestbps INT,
                    chol INT,
                    fbs INT,
                    restecg INT,
                    thalach INT,
                    exang INT,
                    oldpeak FLOAT,
                    slope INT,
                    ca INT,
                    thal INT,
                    predicted_outcome VARCHAR(255),
                    user_feedback BOOLEAN,
                    feedback_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_multiple (
                    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    itching BOOLEAN,
                    skin_rash TINYINT(1) DEFAULT 0,
                    nodal_skin_eruptions TINYINT(1) DEFAULT 0,
                    continuous_sneezing TINYINT(1) DEFAULT 0, 
                    shivering TINYINT(1) DEFAULT 0, 
                    chills TINYINT(1) DEFAULT 0, 
                    joint_pain TINYINT(1) DEFAULT 0, 
                    stomach_pain TINYINT(1) DEFAULT 0, 
                    acidity TINYINT(1) DEFAULT 0, 
                    ulcers_on_tongue TINYINT(1) DEFAULT 0, 
                    muscle_wasting TINYINT(1) DEFAULT 0, 
                    vomiting TINYINT(1) DEFAULT 0, 
                    burning_micturition TINYINT(1) DEFAULT 0, 
                    spotting_urination TINYINT(1) DEFAULT 0, 
                    fatigue TINYINT(1) DEFAULT 0, 
                    weight_gain TINYINT(1) DEFAULT 0, 
                    anxiety TINYINT(1) DEFAULT 0, 
                    cold_hands_and_feets TINYINT(1) DEFAULT 0, 
                    mood_swings TINYINT(1) DEFAULT 0, 
                    weight_loss TINYINT(1) DEFAULT 0, 
                    restlessness TINYINT(1) DEFAULT 0, 
                    lethargy TINYINT(1) DEFAULT 0, 
                    patches_in_throat TINYINT(1) DEFAULT 0, 
                    irregular_sugar_level TINYINT(1) DEFAULT 0, 
                    cough TINYINT(1) DEFAULT 0, 
                    high_fever TINYINT(1) DEFAULT 0, 
                    sunken_eyes TINYINT(1) DEFAULT 0, 
                    breathlessness TINYINT(1) DEFAULT 0, 
                    sweating TINYINT(1) DEFAULT 0, 
                    dehydration TINYINT(1) DEFAULT 0, 
                    indigestion TINYINT(1) DEFAULT 0, 
                    headache TINYINT(1) DEFAULT 0, 
                    yellowish_skin TINYINT(1) DEFAULT 0, 
                    dark_urine TINYINT(1) DEFAULT 0, 
                    nausea TINYINT(1) DEFAULT 0, 
                    loss_of_appetite TINYINT(1) DEFAULT 0, 
                    pain_behind_the_eyes TINYINT(1) DEFAULT 0, 
                    back_pain TINYINT(1) DEFAULT 0, 
                    constipation TINYINT(1) DEFAULT 0, 
                    abdominal_pain TINYINT(1) DEFAULT 0, 
                    diarrhoea TINYINT(1) DEFAULT 0, 
                    mild_fever TINYINT(1) DEFAULT 0, 
                    yellow_urine TINYINT(1) DEFAULT 0, 
                    yellowing_of_eyes TINYINT(1) DEFAULT 0, 
                    acute_liver_failure TINYINT(1) DEFAULT 0, 
                    fluid_overload TINYINT(1) DEFAULT 0, 
                    swelling_of_stomach TINYINT(1) DEFAULT 0, 
                    swelled_lymph_nodes TINYINT(1) DEFAULT 0, 
                    malaise TINYINT(1) DEFAULT 0, 
                    blurred_and_distorted_vision TINYINT(1) DEFAULT 0, 
                    phlegm TINYINT(1) DEFAULT 0, 
                    throat_irritation TINYINT(1) DEFAULT 0, 
                    redness_of_eyes TINYINT(1) DEFAULT 0, 
                    sinus_pressure TINYINT(1) DEFAULT 0, 
                    runny_nose TINYINT(1) DEFAULT 0, 
                    congestion TINYINT(1) DEFAULT 0, 
                    chest_pain TINYINT(1) DEFAULT 0, 
                    weakness_in_limbs TINYINT(1) DEFAULT 0, 
                    fast_heart_rate TINYINT(1) DEFAULT 0, 
                    pain_during_bowel_movements TINYINT(1) DEFAULT 0, 
                    pain_in_anal_region TINYINT(1) DEFAULT 0, 
                    bloody_stool TINYINT(1) DEFAULT 0, 
                    irritation_in_anus TINYINT(1) DEFAULT 0, 
                    neck_pain TINYINT(1) DEFAULT 0, 
                    dizziness TINYINT(1) DEFAULT 0, 
                    cramps TINYINT(1) DEFAULT 0, 
                    bruising TINYINT(1) DEFAULT 0, 
                    obesity TINYINT(1) DEFAULT 0, 
                    swollen_legs TINYINT(1) DEFAULT 0, 
                    swollen_blood_vessels TINYINT(1) DEFAULT 0, 
                    puffy_face_and_eyes TINYINT(1) DEFAULT 0, 
                    enlarged_thyroid TINYINT(1) DEFAULT 0, 
                    brittle_nails TINYINT(1) DEFAULT 0, 
                    swollen_extremeties TINYINT(1) DEFAULT 0, 
                    excessive_hunger TINYINT(1) DEFAULT 0, 
                    extra_marital_contacts TINYINT(1) DEFAULT 0, 
                    drying_and_tingling_lips TINYINT(1) DEFAULT 0, 
                    slurred_speech TINYINT(1) DEFAULT 0, 
                    knee_pain TINYINT(1) DEFAULT 0, 
                    hip_joint_pain TINYINT(1) DEFAULT 0, 
                    muscle_weakness TINYINT(1) DEFAULT 0, 
                    stiff_neck TINYINT(1) DEFAULT 0, 
                    swelling_joints TINYINT(1) DEFAULT 0, 
                    movement_stiffness TINYINT(1) DEFAULT 0, 
                    spinning_movements TINYINT(1) DEFAULT 0, 
                    loss_of_balance TINYINT(1) DEFAULT 0, 
                    unsteadiness TINYINT(1) DEFAULT 0, 
                    weakness_of_one_body_side TINYINT(1) DEFAULT 0, 
                    loss_of_smell TINYINT(1) DEFAULT 0, 
                    bladder_discomfort TINYINT(1) DEFAULT 0, 
                    foul_smell_of_urine TINYINT(1) DEFAULT 0, 
                    continuous_feel_of_urine TINYINT(1) DEFAULT 0, 
                    passage_of_gases TINYINT(1) DEFAULT 0, 
                    internal_itching TINYINT(1) DEFAULT 0, 
                    toxic_look_typhos TINYINT(1) DEFAULT 0, 
                    depression TINYINT(1) DEFAULT 0, 
                    irritability TINYINT(1) DEFAULT 0, 
                    muscle_pain TINYINT(1) DEFAULT 0, 
                    altered_sensorium TINYINT(1) DEFAULT 0, 
                    red_spots_over_body TINYINT(1) DEFAULT 0, 
                    belly_pain TINYINT(1) DEFAULT 0, 
                    abnormal_menstruation TINYINT(1) DEFAULT 0, 
                    dischromic_patches TINYINT(1) DEFAULT 0, 
                    watering_from_eyes TINYINT(1) DEFAULT 0, 
                    increased_appetite TINYINT(1) DEFAULT 0, 
                    polyuria TINYINT(1) DEFAULT 0, 
                    family_history TINYINT(1) DEFAULT 0, 
                    mucoid_sputum TINYINT(1) DEFAULT 0, 
                    rusty_sputum TINYINT(1) DEFAULT 0, 
                    lack_of_concentration TINYINT(1) DEFAULT 0, 
                    visual_disturbances TINYINT(1) DEFAULT 0, 
                    receiving_blood_transfusion TINYINT(1) DEFAULT 0, 
                    receiving_unsterile_injections TINYINT(1) DEFAULT 0, 
                    coma TINYINT(1) DEFAULT 0, 
                    stomach_bleeding TINYINT(1) DEFAULT 0, 
                    distention_of_abdomen TINYINT(1) DEFAULT 0, 
                    history_of_alcohol_consumption TINYINT(1) DEFAULT 0, 
                    blood_in_sputum TINYINT(1) DEFAULT 0, 
                    prominent_veins_on_calf TINYINT(1) DEFAULT 0, 
                    palpitations TINYINT(1) DEFAULT 0, 
                    painful_walking TINYINT(1) DEFAULT 0, 
                    pus_filled_pimples TINYINT(1) DEFAULT 0, 
                    blackheads TINYINT(1) DEFAULT 0, 
                    scurring TINYINT(1) DEFAULT 0, 
                    skin_peeling TINYINT(1) DEFAULT 0, 
                    silver_like_dusting TINYINT(1) DEFAULT 0, 
                    small_dents_in_nails TINYINT(1) DEFAULT 0, 
                    inflammatory_nails TINYINT(1) DEFAULT 0, 
                    blister TINYINT(1) DEFAULT 0, 
                    red_sore_around_nose TINYINT(1) DEFAULT 0, 
                    yellow_crust_ooze TINYINT(1) DEFAULT 0, 
                    prognosis VARCHAR(255) NOT NULL,
                    user_feedback TINYINT(1) DEFAULT 0, 
                    feedback_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            connection.commit()
            return connection
        else:
            st.error("Failed to connect to the database.")
            return None
    except mysql.connector.Error as err:
        st.error(f"Error inserting feedback: {err}")
        return None


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
    features_columns = train_data.drop('prognosis', axis=1).columns.tolist()
except FileNotFoundError:
    st.error(f"Error loading 'Training.csv' ('{TRAINING_DATA_PATH}'). Please ensure it's in the same directory as the app.")
    st.stop()
except Exception as e:
    st.error(f"An unexpected error occurred while loading 'Training.csv': {e}")
    st.stop()

# Function to predict multiple diseases based on symptoms
def predict_diseases(symptoms_str):
    symptoms = [s.strip().lower().replace(' ', '_') for s in symptoms_str.split(',')]
    input_data = pd.DataFrame(np.zeros((1, len(features_columns)), dtype=int), columns=features_columns)
    for symptom in symptoms:
        if symptom in input_data.columns:
            input_data[symptom] = 1

    prediction_encoded = combined_model.predict(input_data)[0]
    predicted_disease = label_encoder.inverse_transform([prediction_encoded])[0]
    return predicted_disease

# Sidebar for navigation
with st.sidebar:
    st.title("PolyDisease Predictor")
    selected = st.radio("Select Predictor", ["🦠 Multiple Disease Prediction", "🩸 Diabetes Prediction", "❤️ Heart Disease Prediction"])

# Multiple Disease Prediction Page
def add_new_symptom_column(mydb, symptom):
    """Adds a new symptom column to the feedback_multiple table if it doesn't exist."""
    safe_symptom = ''.join(c if c.isalnum() or c == '_' else '_' for c in symptom)
    try:
        with mydb.cursor() as mycursor:
            # Check if the column already exists
            mycursor.execute(f"SHOW COLUMNS FROM feedback_multiple LIKE '{safe_symptom}'")
            result = mycursor.fetchone()
            if not result:
                # Add the new column BEFORE the 'prognosis' column
                mycursor.execute(f"ALTER TABLE feedback_multiple ADD COLUMN {safe_symptom} TINYINT(1) DEFAULT 0 AFTER yellow_crust_ooze")
                mydb.commit()
                print(f"Added new column: {safe_symptom} before prognosis")
            else:
                print(f"Column {safe_symptom} already exists.")
    except mysql.connector.Error as err:
        print(f"Error adding column {safe_symptom}: {err}")

if selected == "🦠 Multiple Disease Prediction":
    st.title("Multiple Disease Prediction using Symptoms")
    symptoms_input = st.text_area("Enter your symptoms (comma-separated):", placeholder="e.g. Itching, Skin Rash, Nodal Skin Eruptions")
    predict_button = st.button("Predict Disease")

    mydb = create_db_connection()  # Corrected function call

    if predict_button and symptoms_input:
        if symptoms_input.strip():
            try:
                # Predict the disease based on the input symptoms
                predicted_disease = predict_diseases(symptoms_input)
                st.success(f"Predicted Disease: {predicted_disease}")

                if mydb and mydb.is_connected():
                    # Process the input symptoms
                    symptoms_list = [s.strip().lower().replace(' ', '_') for s in symptoms_input.split(',')]

                    # Check and add new symptoms as columns
                    for symptom in symptoms_list:
                        if symptom not in features_columns:
                            add_new_symptom_column(mydb, symptom)
                            features_columns.append(symptom)  # Update the features_columns list

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("👍 Correct", key="correct_multiple"):
                            feedback_data = {'prognosis': predicted_disease, 'user_feedback': True}
                            for col in features_columns:
                                feedback_data[col] = 1 if col in symptoms_list else 0

                            columns_to_insert = [col for col in feedback_data.keys() if col not in ['id', 'feedback_timestamp']]
                            columns = ", ".join(columns_to_insert)
                            placeholders = ", ".join(["%s"] * len(columns_to_insert))
                            sql = f"INSERT INTO feedback_multiple ({columns}) VALUES ({placeholders})"
                            values = tuple(feedback_data[col] for col in columns_to_insert)

                            print(f"SQL (Correct): {sql}")
                            print(f"Values (Correct): {values}")

                            try:
                                with mydb.cursor() as mycursor:
                                    mycursor.execute(sql, values)
                                    mydb.commit()
                                    st.success("Thank you for your feedback!")
                                    st.info(f"Data Added - SQL: {sql}")
                                    st.info(f"Data Added - Values: {values}")
                            except mysql.connector.Error as err:
                                st.error(f"Error inserting feedback: {err}")
                                print(f"SQL (Error - Correct): {sql}")
                                print(f"Values (Error - Correct): {values}")

                    with col2:
                        if st.button("👎 Incorrect", key="incorrect_multiple"):
                            st.warning("Thank you for your feedback. Please tell us the correct disease if you know it:")
                            correct_disease_input = st.text_input("Correct Disease (optional):", "", key="correct_disease_multiple")
                            if st.button("Submit Correct Disease", key="submit_correct_multiple"):
                                feedback_data = {
                                    'prognosis': correct_disease_input.strip().capitalize() if correct_disease_input else None,
                                    'user_feedback': False,
                                }
                                for col in features_columns:
                                    feedback_data[col] = 1 if col in symptoms_list else 0

                                columns_to_insert = [col for col in feedback_data.keys() if col not in ['id', 'feedback_timestamp']]
                                columns = ", ".join(columns_to_insert)
                                placeholders = ", ".join(["%s"] * len(columns_to_insert))
                                sql = f"INSERT INTO feedback_multiple ({columns}) VALUES ({placeholders})"
                                values = tuple(feedback_data[col] for col in columns_to_insert)

                                print(f"SQL (Incorrect): {sql}")
                                print(f"Values (Incorrect): {values}")

                                try:
                                    with mydb.cursor() as mycursor:
                                        mycursor.execute(sql, values)
                                        mydb.commit()
                                        st.info("Thank you for the corrected information!")
                                        st.info(f"Data Added - SQL: {sql}")
                                        st.info(f"Data Added - Values: {values}")
                                except mysql.connector.Error as err:
                                    st.error(f"Error inserting feedback: {err}")
                                    print(f"SQL (Error - Incorrect): {sql}")
                                    print(f"Values (Error - Incorrect): {values}")
                elif mydb is None:
                    st.error("Database connection failed.")

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
            input_features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
            diab_prediction = diabetes_model.predict([input_features])
            diab_diagnosis = "The person is diabetic" if diab_prediction[0] == 1 else "The person is not diabetic"
            st.success(diab_diagnosis)

            # --- User Feedback Section for Diabetes ---
            mydb = create_db_connection()
            if mydb and mydb.is_connected():
                mycursor = mydb.cursor()
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Correct", key="correct_diabetes"):
                        sql = "INSERT INTO feedback_diabetes (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, predicted_outcome, user_feedback) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                        val = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, diab_diagnosis, True)
                        try:
                            mycursor.execute(sql, val)
                            mydb.commit()
                            st.success("Thank you for your feedback!")
                            st.info(f"Data Added - SQL: {sql}")
                            st.info(f"Data Added - Values: {val}")
                        except mysql.connector.Error as err:
                            st.error(f"Error inserting feedback: {err}")
                            print(f"SQL: {sql}")
                            print(f"Values: {values}")
                        finally:
                            mycursor.close()
                    else:
                        st.error("Failed to connect to the database for feedback.")
                with col2:
                    if st.button("👎 Incorrect", key="incorrect_diabetes"):
                        st.warning("Thank you for your feedback. Please tell us the correct diagnosis if you know it:")
                        correct_diagnosis_input = st.text_input("Correct Diagnosis (optional):", "", key="correct_diagnosis_diabetes")
                        if st.button("Submit Correct Diagnosis", key="submit_correct_diabetes"):
                            sql = "INSERT INTO feedback_diabetes (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, predicted_outcome, user_feedback, correct_diagnosis) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                            val = (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, diab_diagnosis, False, correct_diagnosis_input.strip().capitalize() if correct_diagnosis_input else None)
                            try:
                                mycursor.execute(sql, val)
                                mydb.commit()
                                st.info("Thank you for the corrected information!")
                                st.info(f"Data Added - SQL: {sql}")
                                st.info(f"Data Added - Values: {val}")
                            except mysql.connector.Error as err:
                                st.error(f"Error inserting corrected feedback: {err}")
                                print(f"SQL: {sql}")
                                print(f"Values: {values}")
                            finally:
                                mycursor.close()
                    else:
                        st.error("Failed to connect to the database for feedback.")
                mydb.close()
            elif mydb is None:
                st.error("Failed to connect to the database for feedback.")

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
        oldpeak_heart = st.number_input("Depression Induced by Exercise", min_value=0.0)
        slope_heart = st.number_input("Slope of the Peak Exercise ST segment", min_value=0, max_value=2)
        ca_heart = st.number_input("Number of Major Vessels colored by Fluoroscopy", min_value=0, max_value=4)
        thal_heart = st.number_input("Thalassemia", min_value=0, max_value=3)

        if st.button("Heart Disease Test Result"):
            input_features = [age_heart, sex_heart, cp_heart, trestbps_heart, chol_heart, fbs_heart, restecg_heart, thalach_heart, exang_heart, oldpeak_heart, slope_heart, ca_heart, thal_heart]
            heart_disease_prediction = heart_disease_model.predict([input_features])
            heart_disease_diagnosis = "The person has heart disease" if heart_disease_prediction[0] == 1 else "The person does not have heart disease"
            st.success(heart_disease_diagnosis)

            # --- User Feedback Section for Heart Disease ---
            mydb = create_db_connection()
            if mydb and mydb.is_connected():
                mycursor = mydb.cursor()
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("👍 Correct", key="correct_heart"):
                        sql = "INSERT INTO feedback_heart_disease (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, predicted_outcome, user_feedback) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                        val = (age_heart, sex_heart, cp_heart, trestbps_heart, chol_heart, fbs_heart, restecg_heart, thalach_heart, exang_heart, oldpeak_heart, slope_heart, ca_heart, thal_heart, heart_disease_diagnosis, True)
                        try:
                            mycursor.execute(sql, val)
                            mydb.commit()
                            st.success("Thank you for your feedback!")
                            st.info(f"Data Added - SQL: {sql}")
                            st.info(f"Data Added - Values: {val}")
                        except mysql.connector.Error as err:
                            st.error(f"Error inserting feedback: {err}")
                            print(f"SQL: {sql}")
                            print(f"Values: {values}")
                        finally:
                            mycursor.close()
                    else:
                        st.error("Failed to connect to the database for feedback.")
                with col2:
                    if st.button("👎 Incorrect", key="incorrect_heart"):
                        st.warning("Thank you for your feedback. Please tell us the correct diagnosis if you know it:")
                        correct_diagnosis_input = st.text_input("Correct Diagnosis (optional):", "", key="correct_diagnosis_heart")
                        if st.button("Submit Correct Diagnosis", key="submit_correct_heart"):
                            sql = "INSERT INTO feedback_heart_disease (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, predicted_outcome, user_feedback, correct_diagnosis) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
                            val = (age_heart, sex_heart, cp_heart, trestbps_heart, chol_heart, fbs_heart, restecg_heart, thalach_heart, exang_heart, oldpeak_heart, slope_heart, ca_heart, thal_heart, heart_disease_diagnosis, False, correct_diagnosis_input.strip().capitalize() if correct_diagnosis_input else None)
                            try:
                                mycursor.execute(sql, val)
                                mydb.commit()
                                st.info("Thank you for the corrected information!")
                                st.info(f"Data Added - SQL: {sql}")
                                st.info(f"Data Added - Values: {val}")
                            except mysql.connector.Error as err:
                                st.error(f"Error inserting corrected feedback: {err}")
                                print(f"SQL: {sql}")
                                print(f"Values: {values}")
                            finally:
                                mycursor.close()
                    else:
                        st.error("Failed to connect to the database for feedback.")
                mydb.close()
            elif mydb is None:
                st.error("Failed to connect to the database for feedback.")
    
    except Exception as e:
        st.error(f"An error occurred while loading the heart disease model: {e}")
