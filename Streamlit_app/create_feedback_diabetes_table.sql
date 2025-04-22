CREATE TABLE IF NOT EXISTS feedback_diabetes (
    id INT AUTO_INCREMENT PRIMARY KEY,
    pregnancies INT NOT NULL,
    glucose INT NOT NULL,
    blood_pressure INT NOT NULL,
    skin_thickness INT NOT NULL,
    insulin INT NOT NULL,
    bmi FLOAT NOT NULL,
    diabetes_pedigree_function FLOAT NOT NULL,
    age INT NOT NULL,
    predicted_outcome VARCHAR(255) NOT NULL,
    user_feedback BOOLEAN NOT NULL DEFAULT 0, 
    feedback_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)