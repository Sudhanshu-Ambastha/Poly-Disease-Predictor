import pandas as pd
import numpy as np
import mysql.connector
from mysql.connector import errorcode
import streamlit as st

def predict_diseases(symptoms_str, features_columns, combined_model, label_encoder):
    symptoms = [s.strip().lower().replace(' ', '_') for s in symptoms_str.split(',')]
    input_data = pd.DataFrame(np.zeros((1, len(features_columns)), dtype=int), columns=features_columns)
    for symptom in symptoms:
        if symptom in input_data.columns:
            input_data[symptom] = 1

    prediction_encoded = combined_model.predict(input_data)[0]
    predicted_disease = label_encoder.inverse_transform([prediction_encoded])[0]
    return predicted_disease

def add_new_symptom_column(mydb, symptom):
    safe_symptom = ''.join(c if c.isalnum() or c == '_' else '_' for c in symptom)
    column_name = f"`{safe_symptom}`"

    try:
        with mydb.cursor() as mycursor:
            print("Checking existing columns in 'feedback_multiple':")
            mycursor.execute("DESCRIBE feedback_multiple;")
            columns_info = mycursor.fetchall()
            existing_db_columns = [col[0] for col in columns_info] 
            print(f"Existing columns in DB: {existing_db_columns}")

            if safe_symptom in existing_db_columns:
                print(f"Column '{safe_symptom}' already exists.")
                return True

            mycursor.execute(
                """
                SELECT COLUMN_NAME
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = DATABASE()
                  AND TABLE_NAME = 'feedback_multiple'
                  AND COLUMN_NAME NOT IN ('id', 'prognosis', 'user_feedback', 'feedback_timestamp', 'correct_diagnosis')
                ORDER BY ORDINAL_POSITION
                """
            )
            symptom_columns = [row[0] for row in mycursor.fetchall()]

            alter_sql = f"ALTER TABLE `feedback_multiple` ADD COLUMN {column_name} BOOLEAN NOT NULL DEFAULT 0"
            if symptom_columns:
                alter_sql += f" AFTER `{symptom_columns[-1]}`" 
            else:
                alter_sql += " AFTER `id`" 

            print(f"Executing SQL: {alter_sql}")
            mycursor.execute(alter_sql)
            mydb.commit()
            print(f"Successfully added column: {safe_symptom}")
            return True

    except mysql.connector.Error as err:
        print(f"MySQL Error: {err}")
        if 'st' in globals():
            st.error(f"MySQL Error: {err}")
        if err.errno == errorcode.ER_TABLE_NOT_FOUND:
            print("Table 'feedback_multiple' not found.")
            if 'st' in globals():
                st.error("Table 'feedback_multiple' not found.")
        elif err.errno == errorcode.ER_DUP_FIELDNAME:
            print(f"Column '{safe_symptom}' already exists (duplicate).")
        return False
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        if 'st' in globals():
            st.error(f"An unexpected error occurred: {e}")
        return False