import streamlit as st
import mysql.connector
from mysql.connector import errorcode

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

                mycursor.execute(sql, tuple(values_to_insert))
                mydb.commit()
                st.success("Feedback submitted!")

        except mysql.connector.Error as err:
            st.error(f"Error inserting feedback: {err}")
        except Exception as e:
            st.error(f"An unexpected error occurred during feedback insertion: {e}")
    else:
        st.error("Database connection failed for feedback.")

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