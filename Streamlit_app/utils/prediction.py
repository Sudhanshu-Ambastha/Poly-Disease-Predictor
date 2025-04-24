import pandas as pd
import numpy as np
import streamlit as st
import mysql.connector
from mysql.connector import errorcode
from utils.model_loading import load_combined_model, load_label_encoder, load_training_data_columns

def predict_diseases(symptoms_str):
    symptoms = [s.strip().lower().replace(' ', '_') for s in symptoms_str.split(',')]
    input_data = pd.DataFrame(np.zeros((1, len(features_columns)), dtype=int), columns=features_columns)
    for symptom in symptoms:
        if symptom in input_data.columns:
            input_data[symptom] = 1

    prediction_encoded = load_combined_model.predict(input_data)[0]
    predicted_disease = load_label_encoder.inverse_transform([prediction_encoded])[0]
    return predicted_disease

def add_new_symptom_column(mydb, symptom):
    """Adds a new symptom column to the feedback_multiple table if it doesn't exist."""
    safe_symptom = ''.join(c if c.isalnum() or c == '_' else '_' for c in symptom)
    try:
        with mydb.cursor() as mycursor:
            mycursor.execute(f"SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME = 'feedback_multiple' AND COLUMN_NAME LIKE '{safe_symptom}'")
            result = mycursor.fetchone()
            if not result:
                # Fetch the list of existing symptom columns to determine where to add the new one
                mycursor.execute("""
                    SELECT COLUMN_NAME
                    FROM INFORMATION_SCHEMA.COLUMNS
                    WHERE TABLE_NAME = 'feedback_multiple'
                    AND COLUMN_NAME NOT IN ('id', 'prognosis', 'user_feedback', 'feedback_timestamp', 'correct_diagnosis')
                    ORDER BY ORDINAL_POSITION
                """)
                symptom_columns_results = mycursor.fetchall()
                symptom_columns = [res[0] for res in symptom_columns_results]

                alter_sql = ""
                if symptom_columns:
                    last_symptom = symptom_columns[-1]
                    alter_sql = f"ALTER TABLE feedback_multiple ADD COLUMN `{safe_symptom}` BOOLEAN NOT NULL DEFAULT 0 AFTER `{last_symptom}`"
                else:
                    alter_sql = f"ALTER TABLE feedback_multiple ADD COLUMN `{safe_symptom}` BOOLEAN NOT NULL DEFAULT 0 AFTER `id`"

                try:
                    mycursor.execute(alter_sql)
                    mydb.commit()
                    print(f"Successfully added column: {safe_symptom}")
                    return True
                except mysql.connector.Error as err:
                    st.error(f"Error altering table to add column '{safe_symptom}': {err}")
                    print(f"MySQL Error (ALTER TABLE): {err}")
                    return False
            else:
                print(f"Column '{safe_symptom}' already exists.")
                return True
    except mysql.connector.Error as err:
        st.error(f"Error checking for column '{safe_symptom}': {err}")
        print(f"MySQL Error (SHOW COLUMNS): {err}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred while adding column '{safe_symptom}': {e}")
        print(f"Unexpected Error (add_new_symptom_column): {e}")
        return False