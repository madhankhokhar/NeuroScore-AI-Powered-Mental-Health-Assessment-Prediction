# Mental Health Prediction Project

## Overview
A web application that combines mental health assessment tools with real-time chat functionality between users and psychiatrists. The application allows users to take mental health assessments and connect with registered psychiatrists for professional guidance.


# Mental Health Prediction Project

## Overview
This project aims to predict depression severity based on various mental health assessment data using machine learning techniques. The project consists of several components, including a Python script for prediction, a dataset containing mental health data, and a Jupyter notebook for exploratory data analysis and model training.

## Features

### For Users/Testers
- Complete mental health assessments including:
  - PHQ (Patient Health Questionnaire) Score
  - GAD (Generalized Anxiety Disorder) Score
  - Epworth Sleepiness Score
  - BMI Calculator
- Receive AI-generated suggestions based on assessment results
- Download assessment results and suggestions as PDF
- Real-time chat with registered psychiatrists
- View list of available psychiatrists

### For Psychiatrists
- Dedicated dashboard to view patient messages
- Real-time chat with users/testers

## Dataset
The dataset used for training the machine learning model can be found at:
[Depression and Anxiety Data](https://www.kaggle.com/datasets/shahzadahmad0402/depression-and-anxiety-data)

For JS logic https://www.youtube.com/watch?v=whEObh8waxg&t=907s this old video really refershes memory 

## Installation
To set up the project, follow these steps:

1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure the Gemini API key:
   - Visit [Google AI Studio](https://aistudio.google.com/app/apikey) to create a new API key.
   - Copy the API key and create a `.env` file in the root directory.
   - Add your API key to the `.env` file:
     ```
     GOOGLE_API_KEY=your_google_api_key_here
     ```

## Usage
0. **Create a virtual environment:**
 ```bash
   # On Windows
   python -m venv venv
   .\venv\Scripts\activate

   # On macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```
   
1. **Run the application:**
   ```bash
   python predict_mental_health.py (make sure you have saved .pkl by running mental_health_model.ipynb script)
   ```

2. **Access the web application:**
   Open your web browser and go to `http://localhost:5000/` or `http://127.0.0.1:5000/`.


3. **Register as either a:**
   - Tester (regular user)
   - Psychiatrist

4. **Log in with your credentials**

5. **Download suggestions:**
   After receiving the suggestions, you can download them as a PDF.

## Key Components
- **predict_mental_health.py**: The main Python script responsible for handling predictions and generating reports.
- **EDA.py**: Contains exploratory data analysis insights about the dataset.
- **mental_health_model.ipynb**: Contains model training and evaluation logic.
- **utlis.py**: Handles API calls and PDF generation.
- **templates/**: Contains HTML templates for user input forms.

## Additional Features
- **PHQ Score Calculation**: A separate page to calculate the PHQ score based on user responses.
- **GAD Score Calculation**: A separate page to calculate the GAD score based on user responses.
- **BMI Calculator**: A separate page to calculate Body Mass Index (BMI) based on user input.

