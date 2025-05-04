import streamlit as st
import mysql.connector
from mysql.connector import errorcode

def insert_feedback_multiple(mydb, symptoms, predicted_disease, user_feedback, correct_diagnosis=None):
    try:
        with mydb.cursor() as mycursor:
            columns = ["prognosis", "user_feedback", "correct_diagnosis"]
            values = [predicted_disease, user_feedback, correct_diagnosis if correct_diagnosis is not None else ""]

            mycursor.execute("DESCRIBE feedback_multiple;")
            existing_columns_info = mycursor.fetchall()
            existing_symptom_columns = [col[0].lower() for col in existing_columns_info if col[0] not in ['id', 'prognosis', 'user_feedback', 'feedback_timestamp', 'correct_diagnosis']]

            for symptom in existing_symptom_columns:
                columns.append(f"`{symptom}`")
                values.append(1 if symptom in [s.lower().replace(' ', '_') for s in symptoms] else 0)

            for symptom in symptoms:
                safe_symptom = f"`{symptom.lower().replace(' ', '_')}`"
                if symptom.lower().replace(' ', '_') not in existing_symptom_columns:
                    try:
                        mycursor.execute(f"ALTER TABLE feedback_multiple ADD COLUMN {safe_symptom} BOOLEAN NOT NULL DEFAULT 0")
                        mydb.commit()
                        columns.append(safe_symptom)
                        values.append(1)
                    except mysql.connector.Error as e:
                        st.error(f"Error adding column {symptom}: {e}")
                        print(f"Error adding column {symptom}: {e}")
                        return False

            placeholders = ", ".join(["%s"] * len(columns))
            sql = f"INSERT INTO feedback_multiple ({', '.join(columns)}) VALUES ({placeholders})"
            mycursor.execute(sql, values)
            mydb.commit()
            return True
    except mysql.connector.Error as err:
        st.error(f"Error inserting feedback: {err}")
        print(f"Error inserting feedback into feedback_multiple: {err}")
        mydb.rollback()
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        print(f"An unexpected error occurred: {e}")
        mydb.rollback()
        return False

def insert_feedback_diabetes(mydb, table_name, input_features, predicted_outcome, user_feedback):
    try:
        mycursor = mydb.cursor()
        sql = f"""
            INSERT INTO {table_name} (
                pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age,
                predicted_outcome, user_feedback
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = input_features + [int(predicted_outcome), user_feedback]
        mycursor.execute(sql, values)
        mydb.commit()
        st.success("Feedback submitted!")
        return True
    except mysql.connector.Error as err:
        st.error(f"Error inserting feedback: {err}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during feedback insertion: {e}")
        return False
    finally:
        if mycursor:
            mycursor.close()

def insert_feedback_heart(mydb, table_name, input_features, predicted_outcome, user_feedback):
    try:
        mycursor = mydb.cursor()
        sql = f"""
            INSERT INTO {table_name} (
                age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal,
                predicted_outcome, user_feedback
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        """
        values = input_features + [int(predicted_outcome), user_feedback]
        mycursor.execute(sql, values)
        mydb.commit()
        st.success("Feedback submitted!")
        return True
    except mysql.connector.Error as err:
        st.error(f"Error inserting feedback: {err}")
        return False
    except Exception as e:
        st.error(f"An unexpected error occurred during feedback insertion: {e}")
        return False
    finally:
        if mycursor:
            mycursor.close()