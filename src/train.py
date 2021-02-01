from azureml.core.run import Run
import argparse
import joblib
import os

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np
import matplotlib.pyplot as plt

from config.config_training import script_params_local

print("defining args")
parser = argparse.ArgumentParser()
parser.add_argument(
    '--param_1',
    type=float,
    dest='param_1',
    default=script_params_local['param_1'])
parser.add_argument(
    "--remote_execution",
    dest="remote_execution",
    default=False,
)
parser.add_argument(
    "--ds",
    dest="azure_name_dataset",
    default=script_params_local['azure_name_dataset']  # path for local debugging
)
parser.add_argument(
    "--path_trained_model",
    dest="path_trained_model",
    default=script_params_local['path_trained_model']  # path for local debugging
)

print("parsing args")
args = parser.parse_args()
param_1 = args.param_1
remote_execution = args.remote_execution
azure_name_dataset = args.source_dataset
path_trained_model = args.path_trained_model


def preprocessing(data, target_name, run):
    # log distinct pregnancy counts
    pregnancies = data.Pregnancies.unique()
    run.log_list('pregnancy categories', pregnancies)

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


def train(X_train, X_test, y_train, y_test, reg, run):
    # Train a logistic regression model
    model = LogisticRegression(C=1 / reg, solver="liblinear").fit(X_train, y_train)

    # calculate accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)

    # Store training result
    run.log('Accuracy', np.float(acc))

    # Plot and log the count of diabetic vs non-diabetic patients
    diabetic_counts = data['Diabetic'].value_counts()
    fig = plt.figure(figsize=(6, 6))
    ax = fig.gca()
    diabetic_counts.plot.bar(ax=ax)
    ax.set_title('Patients with Diabetes')
    ax.set_xlabel('Diagnosis')
    ax.set_ylabel('Patients')
    plt.show()
    run.log_image(name='label distribution', plot=fig)

    # Save the trained model
    os.makedirs(path_trained_model, exist_ok=True)
    joblib.dump(value=model, filename=path_trained_model + 'diabetes_model.pkl')


def main():
    #  get context from run
    run = Run.get_context()

    # Get parameters
    run.log('general_information', 'This is my first test')
    run.log("lr_decay", param_1)

    #  Load Data
    tab_ds = run.input_datasets['diabetes_dataset']
    df = tab_ds.to_pandas_dataframe()

    # Preprocess Data
    X_train_scaled, X_test_scaled, y_train, y_test = preprocessing(df, 'Diabetic', run)

    # Train Data
    train(X_train_scaled, X_test_scaled, y_train, y_test, param_1, run)