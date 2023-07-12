# Written by Anmol Gulati
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score

# Data Collection and Analysis
diabetes_dataset = pd.read_csv('diabetes.csv')  # loading data to pandas dataframe
diabetes_dataset.head()  # outcome 0 means non-diabetic, 1 means diabetic
diabetes_dataset.shape  # no. of rows and cols in dataset
diabetes_dataset.describe()  # gets statistical measures of data

diabetes_dataset['Outcome'].value_counts()  # 500 w outcome 0, 268 w outcome 1
diabetes_dataset.groupby('Outcome').mean()
X = diabetes_dataset.drop(columns='Outcome', axis=1)
Y = diabetes_dataset['Outcome']


# Data Standardization
scaler = StandardScaler()
scaler.fit(X)
standardized_data = scaler.transform(X)
X = standardized_data  # represents data
Y = diabetes_dataset['Outcome']  # represents model

# Train Test Split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, stratify=Y, random_state=2)
# print(X.shape, X_train.shape, X_test.shape) # 614 training data, 154 test data

# Training the Model
classifier = svm.SVC(kernel='linear')
classifier.fit(X_train, Y_train)  # training support vector machine classifier

# Model Evaluation
X_train_prediction = classifier.predict(X_train)
training_data_accuracy = accuracy_score(X_train_prediction, Y_train)
print("Accuracy score of the training data: ", training_data_accuracy)

# accuracy score on the test data
X_test_prediction = classifier.predict(X_test)
test_data_accuracy = accuracy_score(X_test_prediction, Y_test)
print("Accuracy score of the test data: ", test_data_accuracy)

# Making a Predictive System
input_data = (5, 116, 74, 0, 0, 25.6, 0.201, 30)  # Input any line of data to test, don't include the outcome
input_data_as_numpy_array = np.asarray(input_data)
input_data_reshape = input_data_as_numpy_array.reshape(1, -1)

# standardize the input data
std_data = scaler.transform(input_data_reshape)
# print(std_data)
prediction = classifier.predict(std_data)
# print(prediction)

if prediction[0] == 0:
    print('The person is not diabetic')
else:
    print('The person is diabetic')
