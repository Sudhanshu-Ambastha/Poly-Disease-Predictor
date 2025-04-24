import os
import streamlit as st
import mysql.connector
from mysql.connector import errorcode

@st.cache_resource
def create_db_connection():
    try:
        if os.environ.get("STREAMLIT_IS_RUNNING") == "1":
            try:
                connection = mysql.connector.connect(
                    host=st.secrets["host"],
                    user=st.secrets["user"],
                    password=st.secrets["password"],
                    port=st.secrets["port"],
                )
                print("Connected using st.secrets (Deployment)")
            except mysql.connector.Error as err:
                st.error(f"Error connecting to MySQL (Deployment): {err}")
                return None
        else:
            try:
                connection = mysql.connector.connect(
                    host=st.secrets["mysql"]["host"],
                    user=st.secrets["mysql"]["user"],
                    password=st.secrets["mysql"]["password"],
                    port=st.secrets["mysql"]["port"],
                )
                print("Connected using [mysql] in secrets (Local)")
            except KeyError:
                st.error("Error: Missing [mysql] section in your local secrets.toml file.")
                return None
            except mysql.connector.Error as err:
                st.error(f"Error connecting to MySQL (Local): {err}")
                return None

        if connection.is_connected():
            cursor = connection.cursor()
            cursor.execute("CREATE DATABASE IF NOT EXISTS sql12775228")
            cursor.execute("USE sql12775228")

            sql_files = [
                "../sql/create_feedback_diabetes_table.sql",
                "../sql/create_feedback_heart_table.sql",
                "../sql/create_feedback_multiple_table.sql",
            ]
            for sql_file in sql_files:
                sql_file_path = os.path.join(os.path.dirname(__file__), sql_file)
                try:
                    with open(sql_file_path, 'r') as f:
                        sql_script = f.read()
                        for statement in sql_script.split(';\n'):
                            if statement.strip():
                                cursor.execute(statement)
                    connection.commit()
                except FileNotFoundError:
                    st.error(f"Error: SQL file not found at {sql_file_path}")
                    return None
                except mysql.connector.Error as err:
                    st.error(f"Error executing SQL from {sql_file}: {err}")
                    return None

            return connection
        else:
            st.error("Failed to connect to the database.")
            return None
    except mysql.connector.Error as err:
        st.error(f"Error connecting to MySQL: {err}")
        return None

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