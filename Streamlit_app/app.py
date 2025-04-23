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
MODEL_FILE_PATH = os.path.join(os.path.dirname(__file__), './models/CombinedModel.sav')
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(__file__), './models/label_encoder.sav')
TRAINING_DATA_PATH = os.path.join(os.path.dirname(__file__), './models/Training.csv')
DIABETES_MODEL_PATH = os.path.join(os.path.dirname(__file__), './models/DiabetesModel.sav')
HEART_DISEASE_MODEL_PATH = os.path.join(os.path.dirname(__file__), './models/HeartModel.sav')

@st.cache_resource
def create_db_connection():
    try:
        connection = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            port=st.secrets["mysql"]["port"],
        )
        if connection.is_connected():
            cursor = connection.cursor()
            # Create the database if it doesn't exist
            cursor.execute("CREATE DATABASE IF NOT EXISTS poly_disease_predictor")
            cursor.execute("USE poly_disease_predictor")

            # Execute SQL files to create tables
            sql_files = [
                "./sql/create_feedback_diabetes_table.sql",
                "./sql/create_feedback_heart_table.sql",
                "./sql/create_feedback_multiple_table.sql",
            ]
            for sql_file in sql_files:
                sql_file_path = os.path.join(os.path.dirname(__file__), sql_file)
                try:
                    with open(sql_file_path, 'r') as f:
                        sql_script = f.read()
                        for statement in sql_script.split(';\n'): # Split by semicolon and newline
                            if statement.strip():
                                cursor.execute(statement)
                    connection.commit()
                except FileNotFoundError:
                    st.error(f"Error: SQL file not found at {sql_file_path}")
                    return None
                except mysql.connector.Error as err:
                    st.error(f"Error executing SQL from {sql_file}: {err}")
                    print(f"MySQL Error ({sql_file}): {err}")
                    return None

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
                # Read the SQL query from the file to get the last symptom column
                sql_file_path = os.path.join(os.path.dirname(__file__), 'find_last_symptom_col_before_prognosis.sql')
                try:
                    with open(sql_file_path, 'r') as f:
                        get_last_symptom_sql = f.read()
                except FileNotFoundError:
                    print(f"Error: SQL file not found at {sql_file_path}")
                    return

                mycursor.execute(get_last_symptom_sql)
                last_symptom_result = mycursor.fetchone()

                if last_symptom_result and last_symptom_result[0]:
                    last_symptom = last_symptom_result[0]
                    # Add the new column after the last symptom
                    mycursor.execute(f"ALTER TABLE feedback_multiple ADD COLUMN {safe_symptom} BOOLEAN NOT NULL DEFAULT 0 AFTER {last_symptom}")
                    mydb.commit()
                    print(f"Added new column: {safe_symptom} after {last_symptom}")
                else:
                    # Handle the case where there are no symptom columns before 'prognosis'
                    # Add it after 'id' as a fallback
                    mycursor.execute(f"ALTER TABLE feedback_multiple ADD COLUMN {safe_symptom} BOOLEAN NOT NULL DEFAULT 0 AFTER id")
                    mydb.commit()
                    print(f"Added new column: {safe_symptom} after id (no preceding symptom found).")
            else:
                print(f"Column {safe_symptom} already exists.")
    except mysql.connector.Error as err:
        print(f"Error adding column {safe_symptom}: {err}")

def insert_feedback_multiple(mydb, symptoms_list, predicted_disease, user_feedback, correct_disease=None):
    if mydb and mydb.is_connected():
        try:
            with mydb.cursor() as mycursor:
                mycursor.execute("SHOW COLUMNS FROM feedback_multiple")
                columns_data = mycursor.fetchall()
                db_columns = [col[0] for col in columns_data]

                symptom_columns = [col for col in db_columns if col not in ['id', 'prognosis', 'user_feedback', 'feedback_timestamp', 'correct_diagnosis']]
                all_columns = symptom_columns + ['prognosis', 'user_feedback']
                placeholders_arr = ['%s'] * len(all_columns)
                all_placeholders = ', '.join(placeholders_arr)

                sql = f"INSERT INTO feedback_multiple ({', '.join(all_columns)}) VALUES ({all_placeholders})"
                values_to_insert = [1 if col in symptoms_list else 0 for col in symptom_columns] + [predicted_disease, user_feedback]

                if correct_disease:
                    all_columns.append('correct_diagnosis')
                    all_placeholders += ', %s'
                    sql = f"INSERT INTO feedback_multiple ({', '.join(all_columns)}) VALUES ({all_placeholders})"
                    values_to_insert.append(correct_disease.strip().capitalize())

                st.info(f"Attempting to INSERT into columns: {all_columns}")
                st.info(f"Attempting to execute SQL: `{sql}` with values: `{values_to_insert}`")

                mycursor.execute(sql, tuple(values_to_insert)) # Execute with a tuple
                mydb.commit()
                st.success("Feedback submitted for multiple disease prediction!")

        except mysql.connector.Error as err:
            st.error(f"Error inserting feedback for multiple disease: {err}")
            print(f"MySQL Error (Multiple): {err}")
        except Exception as e:
            st.error(f"An unexpected error occurred during multiple disease feedback insertion: {e}")
            print(f"Unexpected Error (Multiple): {e}")
    else:
        st.error("Database connection failed for multiple disease feedback.")

def insert_feedback_diabetes(mydb, table_name, input_features, predicted_outcome, user_feedback):
    """Inserts feedback into the feedback_diabetes table with detailed logging."""
    print("--- insert_feedback_diabetes CALLED ---")
    print(f"Database connection object: {mydb}")
    if mydb and mydb.is_connected():
        print("Database connection is active.")
    else:
        print("Database connection is NOT active!")
        st.error("Database connection is not available in insert_feedback_diabetes.")
        return False

    try:
        mycursor = mydb.cursor()
        sql = f"""
            INSERT INTO {table_name} (
                pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age,
                predicted_outcome, user_feedback
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = input_features + [int(predicted_outcome), user_feedback]  # Ensure predicted_outcome is an integer

        print(f"Executing Diabetes Feedback SQL: {sql}")
        print(f"With values: {values}")

        mycursor.execute(sql, values)
        print("mycursor.execute() COMPLETED.")
        mydb.commit()
        print("mydb.commit() COMPLETED.")
        print("Successfully inserted diabetes feedback.")
        return True
    except mysql.connector.Error as err:
        st.error(f"Error inserting diabetes feedback: {err}")
        print(f"MySQL Error (Diabetes Feedback): {err}")
        print(f"SQL: {sql}")
        print(f"Values: {values}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred in insert_feedback_diabetes: {e}")
        print(f"Unexpected Error (Diabetes Feedback): {e}")
        print(f"SQL: {sql}")
        print(f"Values: {values}")
        return False
    finally:
        if mycursor:
            mycursor.close()
            print("mycursor CLOSED.")
        if mydb and mydb.is_connected():
            print("Database connection STILL active at the end of function.")
        else:
            print("Database connection NOT active at the end of function.")

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
        if mydb and mydb.is_connected():
            st.success("Database connection successful!")
        else:
            st.error("Database connection failed.")
            st.stop()

        st.info("Reached the section before prediction and feedback logic.")

        if predict_button and symptoms_input:
            if symptoms_input.strip():
                try:
                    predicted_disease = predict_diseases(symptoms_input)
                    st.success(f"Predicted Disease: {predicted_disease}")

                    st.session_state['predicted_disease'] = predicted_disease
                    st.session_state['symptoms_list'] = [s.strip().lower().replace(' ', '_') for s in symptoms_input.split(',')]

                    if mydb and mydb.is_connected():
                        for symptom in st.session_state['symptoms_list']:
                            if symptom not in features_columns:
                                add_new_symptom_column(mydb, symptom)
                                features_columns.append(symptom)
                                print(f"features_columns after adding: {features_columns}")

                except Exception as e:
                    st.error(f"An error occurred during prediction: {e}")
                    st.stop()
            else:
                st.warning("Please enter some symptoms.")
                st.stop()

        # Feedback buttons - Moved outside the 'predict_button' block
        if st.session_state.get('predicted_disease'):
            col1, col2 = st.columns(2)

            with col1:
                if st.button("👍 Correct", key="correct_multiple_btn"):
                    st.info("👍 Correct button was clicked!") # Debugging
                    print("Calling insert_feedback_multiple for correct feedback") # Debugging
                    insert_feedback_multiple(mydb, st.session_state.get('symptoms_list'), st.session_state.get('predicted_disease'), True)
                    with mydb.cursor() as mycursor:
                        mycursor.execute("""SELECT * FROM poly_disease_predictor.feedback_multiple ORDER BY feedback_timestamp DESC LIMIT 5""")
                        st.session_state['feedback_data'] = mycursor.fetchall()
                        mydb.commit()

            with col2:
                if st.button("👎 Incorrect", key="incorrect_multiple_btn"):
                    st.info("👎 Incorrect button was clicked!") # Debugging
                    correct_disease_input = st.text_input("Correct Disease (optional):", "", key="correct_disease_multiple_input")
                    if st.button("Submit Correct Disease", key="submit_correct_multiple_btn"):
                        st.info("Submit Correct Disease button was clicked!") # Debugging
                        print("Calling insert_feedback_multiple for incorrect feedback") # Debugging
                        insert_feedback_multiple(mydb, st.session_state.get('symptoms_list'), st.session_state.get('predicted_disease'), False, correct_disease_input)
                        with mydb.cursor() as mycursor:
                            mycursor.execute("""SELECT * FROM poly_disease_predictor.feedback_multiple ORDER BY feedback_timestamp DESC LIMIT 5""")
                            st.session_state['feedback_data'] = mycursor.fetchall()
                            mydb.commit()

        if st.session_state.get('feedback_data'):
            st.subheader("Recent Feedback:")
            try:
                mydb_feedback = create_db_connection()  # Ensure a valid connection
                if mydb_feedback and mydb_feedback.is_connected():
                    with mydb_feedback.cursor() as mycursor_feedback:
                        if mycursor_feedback.description:
                            column_names = [desc[0] for desc in mycursor_feedback.description]
                            df_feedback = pd.DataFrame(st.session_state['feedback_data'], columns=column_names)
                            st.dataframe(df_feedback)
                        else:
                            st.info("No feedback data available yet.")
                else:
                    st.error("Failed to connect to the database to display feedback.")
            except mysql.connector.Error as err:
                st.error(f"Error fetching feedback data: {err}")
                print(f"MySQL Error (Feedback Display): {err}")
            except Exception as e:
                st.error(f"An unexpected error occurred while displaying feedback: {e}")
                print(f"Unexpected Error (Feedback Display): {e}")

# Diabetes Prediction Page
if "diab_diagnosis" not in st.session_state:
    st.session_state.diab_diagnosis = None

if selected == "🩸 Diabetes Prediction":
    st.title("Diabetes Prediction using ML")
    mydb = create_db_connection()

    if not mydb or not mydb.is_connected():
        st.error("Failed to connect to the database for feedback.")

    try:
        diabetes_model = joblib.load(open(DIABETES_MODEL_PATH, 'rb'))

        pregnancies = st.number_input("Number of Pregnancies", min_value=0)
        glucose = st.number_input("Glucose Level", min_value=0)
        blood_pressure = st.number_input("Blood Pressure value", min_value=0)
        skin_thickness = st.number_input("Skin Thickness value", min_value=0)
        insulin = st.number_input("Insulin Level", min_value=0)
        bmi = st.number_input("BMI value", min_value=0.0)
        diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function value", min_value=0.0)
        age = st.number_input("Age of the Person", min_value=0)

        if st.button("Diabetes Test Result", key="diabetes_test_result_button"):
            input_features = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
            diab_prediction = diabetes_model.predict([input_features])
            diab_diagnosis = diab_prediction[0]
            st.session_state.diab_diagnosis = diab_diagnosis

            if diab_diagnosis == 1:
                st.success("Prediction Result: Diabetic")
            else:
                st.success("Prediction Result: Not Diabetic")

        # --- User Feedback Section for Diabetes ---
        if st.session_state.get('diab_diagnosis') is not None:
            col1, col2 = st.columns(2)
            with col1:
                if st.button("👍 Correct", key=f"correct_diabetes_button_{st.session_state.get('diab_diagnosis')}"):
                    print("--- 'Correct' button CLICKED ---") # DEBUG
                    st.info("👍 Correct button clicked (Diabetes)") # Debugging
                    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
                    predicted = st.session_state.diab_diagnosis
                    feedback_value = True
                    print(f"Inserting Correct Feedback - Input Data: {input_data}, Predicted: {predicted}, Feedback: {feedback_value}")
                    if insert_feedback_diabetes(create_db_connection(), "feedback_diabetes", input_data, predicted, feedback_value):
                        st.success("Thank you for your feedback!")
                    else:
                        st.error("Error submitting feedback.")
                    st.info("Feedback processing done (Correct - Diabetes)") # Debugging

            with col2:
                if st.button("👎 Incorrect", key=f"incorrect_diabetes_button_{st.session_state.get('diab_diagnosis')}"):
                    print("--- 'Incorrect' button CLICKED ---") # DEBUG
                    st.info("👎 Incorrect button clicked (Diabetes)") # Debugging
                    input_data = [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]
                    predicted = st.session_state.diab_diagnosis
                    feedback_value = False
                    correct_diagnosis_input = st.text_input("Correct Diagnosis (optional):", "", key=f"correct_diagnosis_diabetes_{st.session_state.get('diab_diagnosis')}")
                    if st.button("Submit Correct Diagnosis", key=f"submit_correct_diabetes_button_{st.session_state.get('diab_diagnosis')}"):
                        print("--- 'Submit Correct Diagnosis' button CLICKED ---") # DEBUG
                        st.info("Submit Correct Diagnosis button clicked (Diabetes)") # Debugging
                        print(f"Inserting Incorrect Feedback - Input Data: {input_data}, Predicted: {predicted}, Feedback: {feedback_value}")
                        if insert_feedback_diabetes(create_db_connection(), "feedback_diabetes", input_data, predicted, feedback_value):
                            st.info("Thank you for the corrected information!")
                        else:
                            st.error("Error submitting feedback.")
                        st.info("Feedback processing done (Incorrect - Diabetes)") # Debugging

    except FileNotFoundError:
        st.error(f"Error loading the diabetes model file ('{DIABETES_MODEL_PATH}'). Please ensure it's in the 'models' directory.")
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the diabetes model: {e}")

    if mydb is None:
        st.error("Failed to connect to the database for feedback.")

# Heart Disease Prediction Page
if selected == "❤️ Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")

    mydb = create_db_connection()
    if mydb.is_connected():
        st.success("Database connection successful (test)!")
        # mydb.close()
    else:
        st.error("Database connection failed (test).")

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
                            print(f"Values: {val}")
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
                                print(f"Values: {val}")
                            finally:
                                mycursor.close()
                    else:
                        st.error("Failed to connect to the database for feedback.")
                mydb.close()

            elif mydb is None:
                st.error("Failed to connect to the database for feedback.")
    
    except Exception as e:
        st.error(f"An error occurred while loading the heart disease model: {e}")