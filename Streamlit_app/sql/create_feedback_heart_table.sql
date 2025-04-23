CREATE TABLE IF NOT EXISTS feedback_heart (
    id INT AUTO_INCREMENT PRIMARY KEY,
    age INT NOT NULL,
    sex BOOLEAN NOT NULL DEFAULT 0,
    cp INT NOT NULL,
    trestbps INT NOT NULL,
    chol INT NOT NULL,
    fbs INT NOT NULL,
    restecg INT NOT NULL,
    thalach INT NOT NULL,
    exang INT NOT NULL,
    oldpeak FLOAT NOT NULL,
    slope INT NOT NULL,
    ca INT NOT NULL,
    thal INT NOT NULL,
    predicted_outcome BOOLEAN NOT NULL DEFAULT 0,
    user_feedback BOOLEAN NOT NULL DEFAULT 0, 
    feedback_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)