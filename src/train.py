from azureml.core.run import Run
import argparse
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import pandas as pd

print(os.getcwd())

print("defining args")
parser = argparse.ArgumentParser()
parser.add_argument(
    '--param_1',
    type=float,
    dest='param_1',
    default=0.2)
parser.add_argument(
    "--remote_execution",
    dest="remote_execution",
    default=False,
)

print("parsing args")
args = parser.parse_args()
param_1 = args.param_1
remote_execution = args.remote_execution


def preprocessing(data, target_name):
    # get feature and target
    X = data.drop([target_name], axis=1)
    y = data[target_name]

    # split data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y)

    # scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train, y_test


def train(X_train, X_test, y_train, y_test, reg):
    # Train a logistic regression model
    model = LogisticRegression(C=1 / reg, solver="liblinear").fit(X_train, y_train)

    # calculate accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)

    if remote_execution:
        run.log('Accuracy', np.float(acc))

        # Save the trained model
        joblib.dump(value=model, filename='./trained_models/model.pkl')


if remote_execution:
    #  get context from run
    run = Run.get_context()

    # Get parameters
    run.log('general_information', 'This is my first test')
    run.log("lr_decay", param_1)

#  Load Data
dataset=pd.read_csv('./data/diabetes.csv', sep=',', decimal='.')

# Preprocess Data
X_train_scaled, X_test_scaled, y_train, y_test = preprocessing(dataset, 'Diabetic')

# Train Data
train(X_train_scaled, X_test_scaled, y_train, y_test, param_1)