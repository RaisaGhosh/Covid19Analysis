from flask import Flask, render_template, request
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import csv


# Step 2: Preprocess the Data (if needed)
def to_num(value):
    if value == 'Yes':
        return 1
    elif value == 'No':
        return 0
    else:
        return -1
    
def format():

    my_array = ['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat','Headache','Running Nose','COVID-19']

    cell_df = pd.read_csv('Covid Dataset.csv')
    cell_df.head()

    cell_df = cell_df.dropna()

    with open('Covid Dataset.csv', 'r') as csvfile:
        csvreader = csv.reader(csvfile)
    
        # Read the header row
        header = next(csvreader)
    
        # Traverse through columns
        for column in header:
            if column in my_array:
                cell_df[column] = cell_df[column].apply(to_num)
            else:
                cell_df = cell_df.drop(column,axis=1)

    print("hi")
    print(cell_df)
    return cell_df


def train(data):
    # Step 3: Split the Data
    X = data.drop('COVID-19', axis=1)  # Features
    y = data['COVID-19']  # Target variable

    # Step 4: Choose a Model
    model = RandomForestClassifier()
    print(X)
    print(y)
    # Step 5: Train the Model
    model.fit(X, y)
    return model


def predict():

    # Step 1: Load the Data
    data = format()
    model = train(data)

    # Get the input values from the form
    breathing_problem = 1
    fever = 1
    dry_cough = 0
    sore_throat = 1
    headache = 1
    running_nose = 0
    
    # Make a prediction using the loaded model
    prediction = model.predict([[breathing_problem,fever,dry_cough,sore_throat,headache,running_nose]])
    
    # Render the prediction result template with the prediction
    print("No" if prediction[0]==0 else "Yes")

if __name__ == '__main__':
    predict()
