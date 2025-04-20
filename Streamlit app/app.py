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
            port=st.secrets["mysql"]["port"],
        )
        if connection.is_connected():
            cursor = connection.cursor()
            # Create the database if it doesn't exist
            cursor.execute("CREATE DATABASE IF NOT EXISTS poly_disease_predictor")
            cursor.execute("USE poly_disease_predictor")

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_diabetes (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    pregnancies INT NOT NULL,
                    glucose INT NOT NULL,
                    blood_pressure INT NOT NULL,
                    skin_thickness INT NOT NULL,
                    insulin INT NOT NULL,
                    bmi FLOAT NOT NULL,
                    diabetes_pedigree_function FLOAT NOT NULL,
                    age INT NOT NULL,
                    predicted_outcome VARCHAR(255) NOT NULL,
                    user_feedback BOOLEAN NOT NULL DEFAULT 0, 
                    feedback_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_heart (
                    id INT AUTO_INCREMENT PRIMARY KEY,
                    age INT NOT NULL,
                    sex BOOLEAN NOT NULL DEFAULT 0,
                    cp INT NOT NULL,
                    trestbps INT NOT NULL,
                    chol INT NOT NULL,
                    fbs INT NOT NULL,
                    restecg INT NOT NULL,
                    thalach INT NOT NULL,
                    exang INT NOT NULL,
                    oldpeak FLOAT NOT NULL,
                    slope INT NOT NULL,
                    ca INT NOT NULL,
                    thal INT NOT NULL,
                    predicted_outcome VARCHAR(255),
                    user_feedback BOOLEAN NOT NULL DEFAULT 0, 
                    feedback_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)

            cursor.execute("""
                CREATE TABLE IF NOT EXISTS feedback_multiple (
                    id INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
                    itching BOOLEAN NOT NULL DEFAULT 0, 
                    skin_rash BOOLEAN NOT NULL DEFAULT 0, 
                    nodal_skin_eruptions BOOLEAN NOT NULL DEFAULT 0, 
                    continuous_sneezing BOOLEAN NOT NULL DEFAULT 0,  
                    shivering BOOLEAN NOT NULL DEFAULT 0,  
                    chills BOOLEAN NOT NULL DEFAULT 0,  
                    joint_pain BOOLEAN NOT NULL DEFAULT 0,  
                    stomach_pain BOOLEAN NOT NULL DEFAULT 0,  
                    acidity BOOLEAN NOT NULL DEFAULT 0,  
                    ulcers_on_tongue BOOLEAN NOT NULL DEFAULT 0,  
                    muscle_wasting BOOLEAN NOT NULL DEFAULT 0,  
                    vomiting BOOLEAN NOT NULL DEFAULT 0,  
                    burning_micturition BOOLEAN NOT NULL DEFAULT 0,  
                    spotting_urination BOOLEAN NOT NULL DEFAULT 0,  
                    fatigue BOOLEAN NOT NULL DEFAULT 0,  
                    weight_gain BOOLEAN NOT NULL DEFAULT 0,  
                    anxiety BOOLEAN NOT NULL DEFAULT 0,  
                    cold_hands_and_feets BOOLEAN NOT NULL DEFAULT 0,  
                    mood_swings BOOLEAN NOT NULL DEFAULT 0,  
                    weight_loss BOOLEAN NOT NULL DEFAULT 0,  
                    restlessness BOOLEAN NOT NULL DEFAULT 0,  
                    lethargy BOOLEAN NOT NULL DEFAULT 0,  
                    patches_in_throat BOOLEAN NOT NULL DEFAULT 0,  
                    irregular_sugar_level BOOLEAN NOT NULL DEFAULT 0,  
                    cough BOOLEAN NOT NULL DEFAULT 0,  
                    high_fever BOOLEAN NOT NULL DEFAULT 0,  
                    sunken_eyes BOOLEAN NOT NULL DEFAULT 0,  
                    breathlessness BOOLEAN NOT NULL DEFAULT 0,  
                    sweating BOOLEAN NOT NULL DEFAULT 0,  
                    dehydration BOOLEAN NOT NULL DEFAULT 0,  
                    indigestion BOOLEAN NOT NULL DEFAULT 0,  
                    headache BOOLEAN NOT NULL DEFAULT 0,  
                    yellowish_skin BOOLEAN NOT NULL DEFAULT 0,  
                    dark_urine BOOLEAN NOT NULL DEFAULT 0,  
                    nausea BOOLEAN NOT NULL DEFAULT 0,  
                    loss_of_appetite BOOLEAN NOT NULL DEFAULT 0,  
                    pain_behind_the_eyes BOOLEAN NOT NULL DEFAULT 0,  
                    back_pain BOOLEAN NOT NULL DEFAULT 0,  
                    constipation BOOLEAN NOT NULL DEFAULT 0,  
                    abdominal_pain BOOLEAN NOT NULL DEFAULT 0,  
                    diarrhoea BOOLEAN NOT NULL DEFAULT 0,  
                    mild_fever BOOLEAN NOT NULL DEFAULT 0,  
                    yellow_urine BOOLEAN NOT NULL DEFAULT 0,  
                    yellowing_of_eyes BOOLEAN NOT NULL DEFAULT 0,  
                    acute_liver_failure BOOLEAN NOT NULL DEFAULT 0,  
                    fluid_overload BOOLEAN NOT NULL DEFAULT 0,  
                    swelling_of_stomach BOOLEAN NOT NULL DEFAULT 0,  
                    swelled_lymph_nodes BOOLEAN NOT NULL DEFAULT 0,  
                    malaise BOOLEAN NOT NULL DEFAULT 0,  
                    blurred_and_distorted_vision BOOLEAN NOT NULL DEFAULT 0,  
                    phlegm BOOLEAN NOT NULL DEFAULT 0,  
                    throat_irritation BOOLEAN NOT NULL DEFAULT 0,  
                    redness_of_eyes BOOLEAN NOT NULL DEFAULT 0,  
                    sinus_pressure BOOLEAN NOT NULL DEFAULT 0,  
                    runny_nose BOOLEAN NOT NULL DEFAULT 0,  
                    congestion BOOLEAN NOT NULL DEFAULT 0,  
                    chest_pain BOOLEAN NOT NULL DEFAULT 0,  
                    weakness_in_limbs BOOLEAN NOT NULL DEFAULT 0,  
                    fast_heart_rate BOOLEAN NOT NULL DEFAULT 0,  
                    pain_during_bowel_movements BOOLEAN NOT NULL DEFAULT 0,  
                    pain_in_anal_region BOOLEAN NOT NULL DEFAULT 0,  
                    bloody_stool BOOLEAN NOT NULL DEFAULT 0,  
                    irritation_in_anus BOOLEAN NOT NULL DEFAULT 0,  
                    neck_pain BOOLEAN NOT NULL DEFAULT 0,  
                    dizziness BOOLEAN NOT NULL DEFAULT 0,  
                    cramps BOOLEAN NOT NULL DEFAULT 0,  
                    bruising BOOLEAN NOT NULL DEFAULT 0,  
                    obesity BOOLEAN NOT NULL DEFAULT 0,  
                    swollen_legs BOOLEAN NOT NULL DEFAULT 0,  
                    swollen_blood_vessels BOOLEAN NOT NULL DEFAULT 0,  
                    puffy_face_and_eyes BOOLEAN NOT NULL DEFAULT 0,  
                    enlarged_thyroid BOOLEAN NOT NULL DEFAULT 0,  
                    brittle_nails BOOLEAN NOT NULL DEFAULT 0,  
                    swollen_extremeties BOOLEAN NOT NULL DEFAULT 0,  
                    excessive_hunger BOOLEAN NOT NULL DEFAULT 0,  
                    extra_marital_contacts BOOLEAN NOT NULL DEFAULT 0,  
                    drying_and_tingling_lips BOOLEAN NOT NULL DEFAULT 0,  
                    slurred_speech BOOLEAN NOT NULL DEFAULT 0,  
                    knee_pain BOOLEAN NOT NULL DEFAULT 0,  
                    hip_joint_pain BOOLEAN NOT NULL DEFAULT 0,  
                    muscle_weakness BOOLEAN NOT NULL DEFAULT 0,  
                    stiff_neck BOOLEAN NOT NULL DEFAULT 0,  
                    swelling_joints BOOLEAN NOT NULL DEFAULT 0,  
                    movement_stiffness BOOLEAN NOT NULL DEFAULT 0,  
                    spinning_movements BOOLEAN NOT NULL DEFAULT 0,  
                    loss_of_balance BOOLEAN NOT NULL DEFAULT 0,  
                    unsteadiness BOOLEAN NOT NULL DEFAULT 0,  
                    weakness_of_one_body_side BOOLEAN NOT NULL DEFAULT 0,  
                    loss_of_smell BOOLEAN NOT NULL DEFAULT 0,  
                    bladder_discomfort BOOLEAN NOT NULL DEFAULT 0,  
                    foul_smell_of_urine BOOLEAN NOT NULL DEFAULT 0,  
                    continuous_feel_of_urine BOOLEAN NOT NULL DEFAULT 0,  
                    passage_of_gases BOOLEAN NOT NULL DEFAULT 0,  
                    internal_itching BOOLEAN NOT NULL DEFAULT 0,  
                    toxic_look_typhos BOOLEAN NOT NULL DEFAULT 0,  
                    depression BOOLEAN NOT NULL DEFAULT 0,  
                    irritability BOOLEAN NOT NULL DEFAULT 0,  
                    muscle_pain BOOLEAN NOT NULL DEFAULT 0,  
                    altered_sensorium BOOLEAN NOT NULL DEFAULT 0,  
                    red_spots_over_body BOOLEAN NOT NULL DEFAULT 0,  
                    belly_pain BOOLEAN NOT NULL DEFAULT 0,  
                    abnormal_menstruation BOOLEAN NOT NULL DEFAULT 0,  
                    dischromic_patches BOOLEAN NOT NULL DEFAULT 0,  
                    watering_from_eyes BOOLEAN NOT NULL DEFAULT 0,  
                    increased_appetite BOOLEAN NOT NULL DEFAULT 0,  
                    polyuria BOOLEAN NOT NULL DEFAULT 0,  
                    family_history BOOLEAN NOT NULL DEFAULT 0,  
                    mucoid_sputum BOOLEAN NOT NULL DEFAULT 0,  
                    rusty_sputum BOOLEAN NOT NULL DEFAULT 0,  
                    lack_of_concentration BOOLEAN NOT NULL DEFAULT 0,  
                    visual_disturbances BOOLEAN NOT NULL DEFAULT 0,  
                    receiving_blood_transfusion BOOLEAN NOT NULL DEFAULT 0,  
                    receiving_unsterile_injections BOOLEAN NOT NULL DEFAULT 0,  
                    coma BOOLEAN NOT NULL DEFAULT 0,  
                    stomach_bleeding BOOLEAN NOT NULL DEFAULT 0,  
                    distention_of_abdomen BOOLEAN NOT NULL DEFAULT 0,  
                    history_of_alcohol_consumption BOOLEAN NOT NULL DEFAULT 0,  
                    blood_in_sputum BOOLEAN NOT NULL DEFAULT 0,  
                    prominent_veins_on_calf BOOLEAN NOT NULL DEFAULT 0,  
                    palpitations BOOLEAN NOT NULL DEFAULT 0,  
                    painful_walking BOOLEAN NOT NULL DEFAULT 0,  
                    pus_filled_pimples BOOLEAN NOT NULL DEFAULT 0,  
                    blackheads BOOLEAN NOT NULL DEFAULT 0,  
                    scurring BOOLEAN NOT NULL DEFAULT 0,  
                    skin_peeling BOOLEAN NOT NULL DEFAULT 0,  
                    silver_like_dusting BOOLEAN NOT NULL DEFAULT 0,  
                    small_dents_in_nails BOOLEAN NOT NULL DEFAULT 0,  
                    inflammatory_nails BOOLEAN NOT NULL DEFAULT 0,  
                    blister BOOLEAN NOT NULL DEFAULT 0,  
                    red_sore_around_nose BOOLEAN NOT NULL DEFAULT 0,  
                    yellow_crust_ooze BOOLEAN NOT NULL DEFAULT 0,  
                    prognosis VARCHAR(255) NOT NULL,
                    user_feedback BOOLEAN NOT NULL DEFAULT 0,  
                    feedback_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)

            connection.commit()
            return connection
        else:
            st.error("Failed to connect to the database.")
            return None
    except mysql.connector.Error as err:
        st.error(f"Error connecting to MySQL: {err}")
        print(f"Error connecting to MySQL: {err}")
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

if 'feedback_data' not in st.session_state:
    st.session_state['feedback_data'] = []

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

def add_new_symptom_column(mydb, symptom):
    """Adds a new symptom column to the feedback_multiple table if it doesn't exist."""
    safe_symptom = ''.join(c if c.isalnum() or c == '_' else '_' for c in symptom)
    try:
        with mydb.cursor() as mycursor:
            # Check if the column already exists
            mycursor.execute(f"SHOW COLUMNS FROM feedback_multiple LIKE '{safe_symptom}'")
            result = mycursor.fetchone()
            if not result:
                # Get the last symptom column before 'prognosis'
                mycursor.execute("""
                    SELECT COLUMN_NAME
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_NAME = 'feedback_multiple'
                    AND TABLE_SCHEMA = DATABASE()
                    AND COLUMN_NAME != 'prognosis'
                    AND ORDINAL_POSITION < (
                        SELECT ORDINAL_POSITION
                        FROM INFORMATION_SCHEMA.COLUMNS
                        WHERE TABLE_NAME = 'feedback_multiple'
                        AND TABLE_SCHEMA = DATABASE()
                        AND COLUMN_NAME = 'prognosis'
                    )
                    ORDER BY ORDINAL_POSITION DESC
                    LIMIT 1
                """)
                last_symptom = mycursor.fetchone()[0]

                # Add the new column after the last symptom
                mycursor.execute(f"ALTER TABLE feedback_multiple ADD COLUMN {safe_symptom} BOOLEAN NOT NULL DEFAULT 0 AFTER {last_symptom}")
                mydb.commit()
                print(f"Added new column: {safe_symptom} after {last_symptom}")
            else:
                print(f"Column {safe_symptom} already exists.")
    except mysql.connector.Error as err:
        print(f"Error adding column {safe_symptom}: {err}")

# Sidebar for navigation
with st.sidebar:
    st.title("PolyDisease Predictor")
    selected = st.radio("Select Predictor", ["🦠 Multiple Disease Prediction", "🩸 Diabetes Prediction", "❤️ Heart Disease Prediction"])

# Multiple Disease Prediction Page
if selected == "🦠 Multiple Disease Prediction":
    st.title("Multiple Disease Prediction using Symptoms")
    symptoms_input = st.text_area("Enter your symptoms (comma-separated):", placeholder="e.g. Itching, Skin Rash, Nodal Skin Eruptions")
    predict_button = st.button("Predict Disease")

    mydb = create_db_connection()
    if mydb.is_connected():
        st.success("Database connection successful (test)!")
        # mydb.close()
    else:
        st.error("Database connection failed (test).")

    if predict_button and symptoms_input:
        if symptoms_input.strip():
            try:
                predicted_disease = predict_diseases(symptoms_input)
                st.success(f"Predicted Disease: {predicted_disease}")

                if mydb and mydb.is_connected():
                    symptoms_list = [s.strip().lower().replace(' ', '_') for s in symptoms_input.split(',')]

                    for symptom in symptoms_list:
                        if symptom not in features_columns:
                            add_new_symptom_column(mydb, symptom)
                            features_columns.append(symptom)
                            print(f"features_columns after adding: {features_columns}")

                    col1, col2 = st.columns(2)

                    with col1:
                        if st.button("👍 Correct", key="correct_multiple"):
                            insert_feedback_multiple(mydb, symptoms_list, predicted_disease, True)
                            with mydb.cursor() as mycursor:
                                mycursor.execute("""SELECT * FROM poly_disease_predictor.feedback_multiple ORDER BY feedback_timestamp DESC LIMIT 5""") # Fetch last 5 entries
                                st.session_state['feedback_data'] = mycursor.fetchall()
                                mydb.commit() # Ensure changes are committed before fetching

                    with col2:
                        if st.button("👎 Incorrect", key="incorrect_multiple"):
                            correct_disease_input = st.text_input("Correct Disease (optional):", "", key="correct_disease_multiple")
                            if st.button("Submit Correct Disease", key="submit_correct_multiple"):
                                insert_feedback_multiple(mydb, symptoms_list, predicted_disease, False, correct_disease_input)
                                with mydb.cursor() as mycursor:
                                    mycursor.execute("""SELECT * FROM poly_disease_predictor.feedback_multiple ORDER BY feedback_timestamp DESC LIMIT 5""") # Fetch last 5 entries
                                    st.session_state['feedback_data'] = mycursor.fetchall()
                                    mydb.commit() # Ensure changes are committed before fetching
                elif mydb is None:
                    st.error("Database connection failed.")

            except Exception as e:
                st.error(f"An error occurred during prediction: {e}")
        else:
            st.warning("Please enter some symptoms.")

    # Display the feedback data
    if st.session_state['feedback_data']:
        st.subheader("Recent Feedback:")
        column_names = [desc[0] for desc in mydb.cursor().description] # Get column names
        df_feedback = pd.DataFrame(st.session_state['feedback_data'], columns=column_names)
        st.dataframe(df_feedback)

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
                
                mydb = create_db_connection()
                if mydb.is_connected():
                    st.success("Database connection successful (test)!")
                    # mydb.close()
                else:
                    st.error("Database connection failed (test).")

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
                        sql = "INSERT INTO feedback_heart (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, predicted_outcome, user_feedback) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
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
                            sql = "INSERT INTO feedback_heart (age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal, predicted_outcome, user_feedback, correct_diagnosis) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"
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
                
                mydb = create_db_connection()
                if mydb.is_connected():
                    st.success("Database connection successful (test)!")
                    # mydb.close()
                else:
                    st.error("Database connection failed (test).")

            elif mydb is None:
                st.error("Failed to connect to the database for feedback.")
    
    except Exception as e:
        st.error(f"An error occurred while loading the heart disease model: {e}")