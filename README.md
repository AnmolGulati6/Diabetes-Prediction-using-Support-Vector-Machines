# Diabetes Prediction using Support Vector Machines

This project aims to predict the likelihood of diabetes in individuals based on various health-related factors. It utilizes the Support Vector Machines (SVM) algorithm for classification. By training the model on a dataset of diabetes-related features, it can make predictions on new, unseen data.

## Project Overview
The main objective of this project is to develop a predictive model for diabetes detection. It involves the following steps:

1. Data Collection and Analysis: The dataset containing information on pregnancies, glucose levels, blood pressure, skin thickness, insulin levels, BMI, diabetes pedigree function, age, and outcome (0 for non-diabetic and 1 for diabetic) is loaded and analyzed.

2. Data Standardization: The input features are standardized using the StandardScaler from scikit-learn to ensure uniformity and improve model performance.

3. Train-Test Split: The dataset is split into training and testing sets for model evaluation. The training set is used to train the SVM classifier.

4. Model Training: The SVM classifier with a linear kernel is trained on the standardized training data.

5. Model Evaluation: The accuracy scores of the trained model are calculated for both the training and testing data to assess its performance.

6. Making Predictions: The trained model is used to predict the diabetes outcome for new input data. An example input is provided, and the corresponding prediction is displayed.

## Getting Started
To run this project locally, follow these steps:

1. Clone the repository:
```
git clone https://github.com/your-username/diabetes-prediction.git
```

2. Install the required dependencies:
```
pip install numpy pandas scikit-learn
```

3. Prepare the dataset:
   - Download the diabetes.csv dataset and place it in the project directory.
   - Ensure that the dataset has the correct column names: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, Outcome.

4. Run the script:
```
python main.py
```

## Results
The accuracy scores of the trained model on the training and testing data are displayed. Additionally, a predictive system is implemented to make predictions on new input data.

## Dataset
The diabetes dataset used for this project contains information on various health-related factors and their association with diabetes. The dataset is in CSV format and consists of the following columns: Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age, and Outcome.

## Conclusion
The SVM-based diabetes prediction model developed in this project demonstrates promising accuracy scores on both the training and testing data. This model can be a valuable tool for identifying individuals at risk of diabetes, thereby enabling timely intervention and medical assistance.

**License:** This project is licensed under the [MIT License](LICENSE).
