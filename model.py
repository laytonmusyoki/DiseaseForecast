import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df_main = pd.read_csv('/home/layton/Desktop/machine_learning/diseases_training.csv')

# Preprocess your data
df_main['diagnosis'] = df_main['diagnosis'].round().astype(int)

# Dataset snapshot
df_main

selected_symptoms = ["itching", "skin_rash", "continuous_sneezing", "chills", "joint_pain", "vomiting", "fatigue", "weight_loss", "cough", "high_fever", "headache", "yellowish_skin", "prognosis", "diagnosis"]

df = df_main[selected_symptoms]

# Use .loc for assigning values
df.loc[:, 'diagnosis'] = df['diagnosis'].round()

# Dataset snapshot after modification
df

# Separating the features from the target
y = df['diagnosis']
X = df.drop(['diagnosis', 'prognosis'], axis=1)

# Splitting the data into training and testing data
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Creating the model
model = svm.SVC(kernel='linear')

# Training the model
model.fit(x_train, y_train)

# Testing the model
y_pred = model.predict(x_test)

# Accuracy of the model
accuracy_score(y_test, y_pred)

# Saving the model
pickle.dump(model, open('model.pkl', 'wb'))
