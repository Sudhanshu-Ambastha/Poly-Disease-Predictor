import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle

# Function to predict diseases based on symptoms
def predict_diseases(symptoms, features, rf):
    # Creating input data for the model
    input_data = [0] * len(features.columns)
    for symptom in symptoms:
        if symptom in features.columns:
            index = features.columns.get_loc(symptom)
            input_data[index] = 1

    # Reshaping the input data
    input_data = pd.DataFrame([input_data], columns=features.columns)

    # Generating predictions
    predictions = rf.predict(input_data)
    return predictions

# Set the paths for your models
base_path = 'C:\\Users\\sudha\\OneDrive\\Documents\\GitHub\\combined-disease-prediction-test'
diabetes_model_path = 'models/diabetes_model.pkl'
heart_disease_model_path = 'models/heart_model.pkl'

# Loading the saved models
diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))
heart_disease_model = pickle.load(open(heart_disease_model_path, 'rb'))

# Sidebar for navigation
with st.sidebar:
    st.title("PolyDisease Predictor")
    selected = st.radio("Select Predictor", ["🩸 Diabetes Prediction", "❤️ Heart Disease Prediction", "🦠 Multiple Disease Prediction"])


# Multiple Disease Prediction Page
if selected == "🦠 Multiple Disease Prediction":
    st.title("Multiple Disease Prediction using Symptoms")

    # Load data
    train_data = pd.read_csv('C:\\Users\\sudha\\OneDrive\\Documents\\GitHub\\combined-disease-prediction-test\\Training.csv')
    test_data = pd.read_csv('C:\\Users\\sudha\\OneDrive\\Documents\\GitHub\\combined-disease-prediction-test\\Testing.csv')

    # Split data into features and target variable
    features = train_data.drop('prognosis', axis=1)
    target = train_data['prognosis']

    # Create RandomForestClassifier
    rf = RandomForestClassifier(n_estimators=100)

    # Train the model
    rf.fit(features, target)

    # User input for symptoms
    symptoms = st.text_input("Enter Symptoms (comma-separated)")

    # Initialize the result
    diagnosis = ''

    # Create a button to check symptoms
    if st.button("Check Symptoms"):
        # Call the predict_diseases function with user input
        diseases = predict_diseases(symptoms.split(","), features, rf)
        st.success(f"Predicted Diseases: {', '.join(diseases)}")

# Diabetes Prediction Page
if selected == "🩸 Diabetes Prediction":
    st.title("Diabetes Prediction using ML")

    # Load data
    diabetes_model_path = 'models/diabetes_model.sav'
    diabetes_model = pickle.load(open(diabetes_model_path, 'rb'))

    # Input fields
    pregnancies = st.number_input("Number of Pregnancies")
    glucose = st.number_input("Glucose Level")
    blood_pressure = st.number_input("Blood Pressure value")
    skin_thickness = st.number_input("Skin Thickness value")
    insulin = st.number_input("Insulin Level")
    bmi = st.number_input("BMI value")
    diabetes_pedigree_function = st.number_input("Diabetes Pedigree Function value")
    age = st.number_input("Age of the Person")

    # Debugging line: Add this line to print out variable values
    st.write("Debug: pregnancies =", pregnancies)

    # Prediction button
    if st.button("Diabetes Test Result"):
        diab_prediction = diabetes_model.predict(
            [[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]]
        )
        diab_diagnosis = "The person is diabetic" if diab_prediction[0] == 1 else "The person is not diabetic"
        st.success(diab_diagnosis)

# Heart Disease Prediction Page
if selected == "❤️ Heart Disease Prediction":
    st.title("Heart Disease Prediction using ML")

    # Load data
    heart_disease_model_path = 'models/heart_disease_model.sav'
    heart_disease_model = pickle.load(open(heart_disease_model_path, 'rb'))

    # Input fields
    age = st.number_input("Age")
    sex = st.number_input("Sex")
    cp = st.number_input("Chest Pain types")
    trestbps = st.number_input("Resting Blood Pressure")
    chol = st.number_input("Serum Cholestoral in mg/dl")
    fbs = st.number_input("Fasting Blood Sugar > 120 mg/dl")
    restecg = st.number_input("Resting Electrocardiographic results")
    thalach = st.number_input("Maximum Heart Rate achieved")
    exang = st.number_input("Exercise Induced Angina")
    oldpeak = st.number_input("ST depression induced by exercise")
    slope = st.number_input("Slope of the peak exercise ST segment")
    ca = st.number_input("Major vessels colored by flourosopy")
    thal = st.number_input("thal: 0 = normal; 1 = fixed defect; 2 = reversable defect")

    # Prediction button
    if st.button("Heart Disease Test Result"):
        heart_prediction = heart_disease_model.predict(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        )
        heart_diagnosis = "The person is having heart disease" if heart_prediction[0] == 1 else "The person does not have any heart disease"
        st.success(heart_diagnosis)