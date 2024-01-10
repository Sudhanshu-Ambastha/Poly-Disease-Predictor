import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

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

# Function to predict diabetes based on symptoms
def predict_diabetes(symptoms, features, svm_classifier, scaler):
    # Creating input data for the model
    input_data = [0] * len(features.columns)
    for symptom in symptoms:
        if symptom in features.columns:
            index = features.columns.get_loc(symptom)
            input_data[index] = 1

    # Reshaping the input data
    input_data = pd.DataFrame([input_data], columns=features.columns)

    # Standardizing the input data
    standardized_data = scaler.transform(input_data)
    
    # Generating predictions
    predictions = svm_classifier.predict(standardized_data)
    return predictions

# Sidebar for navigation
with st.sidebar:
    st.title("PolyDisease Predictor")
    selected = st.radio("Select Predictor", ["🩸 Diabetes Prediction", "❤️ Heart Disease Prediction", "🦠 Multiple Disease Prediction"])

# Load data
if selected == "🩸 Diabetes Prediction" or selected == "❤️ Heart Disease Prediction":
    # Load data for diabetes and heart disease prediction
    diabetes_data = pd.read_csv('C:\\Users\\sudha\\OneDrive\\Documents\\GitHub\\combined-disease-prediction-test\\diabetes.csv')  # Update with your diabetes dataset path
    heart_disease_data = pd.read_csv('/content/heart.csv')  # Update with your heart disease dataset path

    # Training SVM model for diabetes prediction
    X_diabetes = diabetes_data.drop(columns='Outcome', axis=1)
    Y_diabetes = diabetes_data['Outcome']
    scaler_diabetes = StandardScaler()
    scaler_diabetes.fit(X_diabetes)
    X_diabetes_standardized = scaler_diabetes.transform(X_diabetes)

    X_diabetes_train, X_diabetes_test, Y_diabetes_train, Y_diabetes_test = train_test_split(X_diabetes_standardized, Y_diabetes, test_size=0.2, stratify=Y_diabetes, random_state=2)

    svm_classifier_diabetes = SVC(kernel='linear')
    svm_classifier_diabetes.fit(X_diabetes_train, Y_diabetes_train)

    # Training RandomForest model for heart disease prediction
    X_heart = heart_disease_data.drop(columns='target', axis=1)
    Y_heart = heart_disease_data['target']
    X_heart_train, X_heart_test, Y_heart_train, Y_heart_test = train_test_split(X_heart, Y_heart, test_size=0.2, stratify=Y_heart, random_state=8)

    rf_classifier_heart = RandomForestClassifier(n_estimators=100)
    rf_classifier_heart.fit(X_heart_train, Y_heart_train)

# Diabetes Prediction Page
if selected == "🩸 Diabetes Prediction":
    st.title("Diabetes Prediction using SVM")

    # User input for symptoms
    symptoms_diabetes = st.text_input("Enter Symptoms (comma-separated)")

    # Initialize the result
    diagnosis_diabetes = ''

    # Create a button to check symptoms
    if st.button("Check Symptoms"):
        # Call the predict_diabetes function with user input
        diabetes_predictions = predict_diabetes(symptoms_diabetes.split(","), X_diabetes, svm_classifier_diabetes, scaler_diabetes)
        diagnosis_diabetes = "The person is diabetic" if diabetes_predictions[0] == 1 else "The person is not diabetic"
        st.success(diagnosis_diabetes)

# Heart Disease Prediction Page
if selected == "❤️ Heart Disease Prediction":
    st.title("Heart Disease Prediction using RandomForest")

    # User input for symptoms
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
        heart_prediction = rf_classifier_heart.predict(
            [[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]]
        )
        heart_diagnosis = "The person is having heart disease" if heart_prediction[0] == 1 else "The person does not have any heart disease"
        st.success(heart_diagnosis)

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
    symptoms_multiple = st.text_input("Enter Symptoms (comma-separated)")

    # Initialize the result
    diagnosis_multiple = ''

    # Create a button to check symptoms
    if st.button("Check Symptoms"):
        # Call the predict_diseases function with user input
        diseases_multiple = predict_diseases(symptoms_multiple.split(","), features, rf)
        diagnosis_multiple = f"Predicted Diseases: {', '.join(diseases_multiple)}"
        st.success(diagnosis_multiple)
