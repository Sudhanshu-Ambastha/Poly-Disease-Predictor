import os
import streamlit as st
import mysql.connector

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

def insert_feedback_diabetes(cursor, db_connection, table, form_data, predicted_outcome, user_feedback, correct_diagnosis=None):
    """Insert feedback into the database."""
    sql = f"""
        INSERT INTO {table} 
        (pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age, predicted_outcome, user_feedback, correct_diagnosis)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
    """
    val = (*form_data, predicted_outcome, user_feedback, correct_diagnosis)
    try:
        cursor.execute(sql, val)
        db_connection.commit()
        return True
    except mysql.connector.Error as err:
        print(f"Error: {err}")
        return False