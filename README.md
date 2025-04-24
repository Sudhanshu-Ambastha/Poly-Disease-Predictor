# PolyDisease Predictor

PolyDisease Predictor is a Streamlit web application that uses machine learning models to predict various diseases, including diabetes, heart disease, and multiple diseases based on user-provided symptoms and health parameters. The application also incorporates user feedback to improve prediction accuracy

## Usage

**Prerequisites:**

  * Python 3.6+
  * Anaconda or Miniconda (recommended)
  * MySQL Server (if you intend to use the database features)

**Setup:**

1.  **Clone the Repository:**

    ```bash
    git clone https://github.com/Sudhanshu-Ambastha/Poly-Disease-Predictor.git
    cd Poly-Disease-Predictor
    ```

2.  **Create a Virtual Environment (Recommended):**

    ```bash
    conda create -n polydisease python=3.9
    conda activate polydisease
    ```

3.  **Install Dependencies:**

    ```bash
    pip install -r Streamlit_app/requirements.txt
    ```
    to save dependencies:
    ```bash
    pip freeze > requirements.txt
    ```

4.  **Database Setup (Optional):**

      * Ensure your MySQL server is running.

      * Create a `.streamlit/secrets.toml` file in your project directory.  This file should contain your database credentials.  See the provided `tomlStruct.txt` for the expected structure.  Example:

        ```toml
        [mysql]
        host = "your_host"
        user = "your_user"
        password = "your_password"
        port = your_port
        ```

5.  **Run the Application:**

    ```bash
    streamlit run Streamlit_app/app.py
    ```

6.  **Access the Application:**

    Open your web browser and navigate to the URL provided in the terminal (usually `http://localhost:8501`).

## Access the Application:

Open your web browser and navigate to the provided URL (usually http://localhost:8501).

## Features

  * **Multiple Disease Prediction:** Predicts potential diseases based on a comma-separated list of symptoms.  Allows user feedback (correct/incorrect) to improve the model.  Dynamically adds new symptoms to the database.
  * **Diabetes Prediction:** Predicts the likelihood of diabetes based on health parameters (pregnancies, glucose, blood pressure, etc.).  Collects user feedback.
  * **Heart Disease Prediction:** Predicts the likelihood of heart disease based on cardiovascular health parameters. Collects user feedback.

## How to Use

1.  Select the desired predictor tab from the sidebar.
2.  Enter the required information (symptoms, health parameters).
3.  Click the "Predict" or "Test Result" button.
4.  Provide feedback on the prediction to help improve the model.

## Data Sources

- Diabetes dataset: Kaggle Diabetes Dataset
  - [diabetes.csv](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)
- Heart disease dataset: Kaggle Heart Disease Dataset
  - [heart.csv](https://www.kaggle.com/code/desalegngeb/heart-disease-predictions/input)
- Multiple disease dataset: Custom dataset used for training and testing.
  - [training. csv](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)

## Important Notes

  * **File Paths:** The application uses relative file paths.  Ensure the `models` directory and `.sql` files are in the correct locations relative to the `app.py` file.
  * **Database Connection:** If you intend to use the feedback features, ensure your MySQL database is set up correctly and the credentials are in the `.streamlit/secrets.toml` file.
  * **Model Files:** The application relies on pre-trained machine learning models (`.sav` files) located in the `models` directory.  Ensure these files are present.
  * **Dependencies:** All required Python packages are listed in `Streamlit_app/requirements.txt`.

**Check out the video related to it**

[![Poly Disease Predictor](https://img.youtube.com/vi/G7AvMkZ0VGM/0.jpg)](https://www.youtube.com/watch?v=G7AvMkZ0VGM&t=1s "Poly Disease Predictor")

**Update:**
This project was created to address challenges faced in earlier models, making it easy to run directly via VS Code, especially when opened through GitHub Desktop. With the presence of necessary files, the project can now be executed seamlessly.

Please checkout the deployed working model here [Poly Disease Predictor](https://poly-disease-predictor.streamlit.app/)

Feel free to contribute and enhance the application!

For contributors looking to integrate a MySQL backend, please refer to the [`tomlStruct.txt`](./.streamlit/tomlStruct.txt) file for the expected structure of the TOML configuration file required for database connection details

## Contributors

<table>
    <tr>
        <td align="center">
        <a href="http://github.com/Sudhanshu-Ambastha">
            <img src="https://avatars.githubusercontent.com/u/135802131?v=4" width="100px;" alt=""/>
            <br />
            <sub><b>Sudhanshu Ambastha </b></sub>
        </a>
        <br />
    </td>
    <td align="center">
        <a href="https://github.com/Vishwas567917">
            <img src="https://avatars.githubusercontent.com/u/139749696?s=100&v=4" width="100px;" alt=""/>
            <br />
            <sub><b>Parth Shrivastava</b></sub>
        </a>
        <br />
    </td>
    <td align="center">
        <a href="https://github.com/Shrivatsa-Sharan-Garg">
            <img src="https://avatars.githubusercontent.com/u/179140208?v=4" width="100px;" alt=""/>
            <br />
            <sub><b>Shrivatsa Sharan Garg</b></sub>
        </a>
        <br/>
    </td>
    </tr>
</table>
