# PolyDisease Predictor

PolyDisease Predictor is a Streamlit web application that allows users to predict various diseases, including diabetes, heart disease, and multiple diseases based on provided symptoms and health parameters.

## Usage

**Clone Repository:**

```bash
git clone https://github.com/Sudhanshu-Ambastha/Poly-Disease-Predictor.git
```

```
cd Streamlit_app
```

**Install Dependencies:**
This command is used to generate a `requirements.txt` file based on the packages currently installed in your active Python environment

```
pip freeze > requirements.txt
```

The command to install dependencies from a `requirements.txt` file

```
pip install -r Streamlit_app/requirements.txt
```

**Run the Application:**

Open Anaconda Command Prompt and run the following command:

```bash
python -m streamlit run "Streamlit_app/app.py"
```

## Access the Application:

Open your web browser and navigate to the provided URL (usually http://localhost:8501).

## Features

Diabetes Prediction
Predicts whether a person is diabetic or not based on provided health parameters.
Heart Disease Prediction
Predicts whether a person has heart disease or not based on cardiovascular health parameters.
Multiple Disease Prediction
Predicts multiple diseases using symptoms provided by the user.

## How to Use

Select the disease predictor from the sidebar.
Follow the instructions and input the required information.
Click the corresponding button to get the prediction result.

## Data Sources

- Diabetes dataset: Kaggle Diabetes Dataset
  - [diabetes.csv](https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset)
- Heart disease dataset: Kaggle Heart Disease Dataset
  - [heart.csv](https://www.kaggle.com/code/desalegngeb/heart-disease-predictions/input)
- Multiple disease dataset: Custom dataset used for training and testing.
[training. csv](https://www.kaggle.com/datasets/kaushil268/disease-prediction-using-machine-learning)

## Note

This project is designed for easy accessibility without downloading the files. Users can run it directly via VS Code by changing the terminal path to make it easily accessible. If users encounter issues like "file not found," ensure the terminal path is correctly set within VS Code.

- **Ensure Python and Anaconda are Installed:**
  Make sure to have Python and Anaconda installed.

- **Internet Connection Required:**
  Internet connection is required to run the application.

- **GitHub Repository Link:**
  Access the GitHub repository for Combined Disease Prediction Bot: [Combined Disease Prediction Bot](https://github.com/Sudhanshu-Ambastha/Combined-Disease-Prediction-Bot)

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