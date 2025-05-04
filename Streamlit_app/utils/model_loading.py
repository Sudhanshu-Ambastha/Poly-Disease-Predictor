import os
import joblib
import pandas as pd
import streamlit as st
import pickle

MODEL_FILE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'CombinedModel.sav')
LABEL_ENCODER_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'label_encoder.sav')
TRAINING_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'Training.csv')
DIABETES_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'DiabetesModel.sav')
HEART_DISEASE_MODEL_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'HeartModel.sav')

@st.cache_resource
def load_combined_model():
    try:
        combined_model = joblib.load(MODEL_FILE_PATH)
        return combined_model
    except FileNotFoundError:
        st.error(f"Error loading the combined model file ('{MODEL_FILE_PATH}'). Please ensure it's in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the combined model: {e}")
        st.stop()

@st.cache_resource
def load_label_encoder():
    try:
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        return label_encoder
    except FileNotFoundError:
        st.error(f"Error loading the LabelEncoder file ('{LABEL_ENCODER_PATH}'). Please ensure it's in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the LabelEncoder: {e}")
        st.stop()

@st.cache_resource
def load_training_data_columns():
    try:
        train_data = pd.read_csv(TRAINING_DATA_PATH)
        features_columns = train_data.drop('prognosis', axis=1).columns.tolist()
        return train_data, features_columns
    
    except FileNotFoundError:
        st.error(f"Error loading 'Training.csv' ('{TRAINING_DATA_PATH}'). Please ensure it's in the same directory as the app.")
        st.stop()
    
    except Exception as e:
        st.error(f"An unexpected error occurred while loading 'Training.csv': {e}")
        st.stop()

@st.cache_resource
def load_diabetes_model():
    try:
        diabetes_model = joblib.load(open(DIABETES_MODEL_PATH, 'rb'))
        return diabetes_model
    except FileNotFoundError:
        st.error(f"Error loading the diabetes model file ('{DIABETES_MODEL_PATH}'). Please ensure it's in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the diabetes model: {e}")
        st.stop()

@st.cache_resource
def load_heart_disease_model():
    try:
        heart_disease_model = pickle.load(open(HEART_DISEASE_MODEL_PATH, 'rb'))
        return heart_disease_model
    except FileNotFoundError:
        st.error(f"Error loading the heart disease model file ('{HEART_DISEASE_MODEL_PATH}'). Please ensure it's in the correct directory.")
        st.stop()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading the heart disease model: {e}")
        st.stop()

import mysql.connector
import streamlit as st
import os

SQL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sql')

SQL_FILES = [
    "create_feedback_diabetes_table.sql",
    "create_feedback_heart_table.sql",
    "create_feedback_multiple_table.sql"
]

@st.cache_resource
def create_db_connection():
    """
    Establishes a connection to the MySQL database and creates the necessary tables
    if they don't exist by executing SQL scripts.
    """
    try:
        connection = mysql.connector.connect(
            host=st.secrets["mysql"]["host"],
            user=st.secrets["mysql"]["user"],
            password=st.secrets["mysql"]["password"],
            port=st.secrets["mysql"]["port"],
            database="sql12775228"
        )
        if connection.is_connected():
            cursor = connection.cursor()
            st.info("Successfully connected to the MySQL database.")

            for sql_file in SQL_FILES:
                file_path = os.path.join(SQL_DIR, sql_file)
                try:
                    with open(file_path, 'r') as f:
                        sql_script = f.read()
                        cursor.execute(sql_script)
                        connection.commit()
                        st.info(f"Successfully executed SQL script: {sql_file}")
                except FileNotFoundError:
                    st.error(f"SQL script not found: {sql_file}")
                except mysql.connector.Error as err:
                    st.error(f"Error executing SQL script '{sql_file}': {err}")
                    connection.rollback()
                except Exception as e:
                    st.error(f"An unexpected error occurred while processing '{sql_file}': {e}")
                    connection.rollback()

            cursor.close()
            return connection
        else:
            st.error("Failed to connect to the database.")
            return None
    except mysql.connector.Error as err:
        st.error(f"Error connecting to MySQL: {err}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred during database connection: {e}")
        return None
