import argparse
import joblib
import os

from azureml.core.run import Run

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import numpy as np

from config.config_training import script_params_local

print("defining args")
parser = argparse.ArgumentParser()
parser.add_argument(
    '--reg_param',
    type=float,
    dest='reg_param',
    default=script_params_local['param_1'])
parser.add_argument(
    "--remote_execution",
    dest="remote_execution",
    default=False,
)
parser.add_argument(
    "--path_trained_model",
    dest="path_trained_model",
    default=script_params_local['path_trained_model']  # path for local debugging
)
parser.add_argument(
    "--in_folder"
    ,dest='in_folder'
    ,default=script_params_local['path_local_data']  # path for local debugging
)


print("parsing args")
args = parser.parse_args()
reg = arg.reg_param
remote_execution = args.remote_execution
path_trained_model = args.path_trained_model
in_folder = args.in_folder


def train(data, reg, run):
    
    # get feature and target
    X = data.drop(['Diabetic'], axis=1)
    y = data[target_name]

    # split data 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, stratify=y)

    # scale data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train a logistic regression model
    model = LogisticRegression(C=1 / reg, solver="liblinear").fit(X_train, y_train)

    # calculate accuracy
    y_hat = model.predict(X_test)
    acc = np.average(y_hat == y_test)

    # Store training result
    run.log('Accuracy', np.float(acc))

    # Save the trained model
    os.makedirs(path_trained_model, exist_ok=True)
    joblib.dump(value=model, filename=path_trained_model + 'diabetes_model.pkl')


def main(data):
    #  get context from run
    run = Run.get_context()

    # Get parameters
    run.log('general_information', 'This is my first test')
    run.log("lr_decay", param_1)
    
    # Get preprocessed data
    input_path = os.path.join(in_folder, 'prepped_data.csv')
    data = pd.read_csv(input_path)

    # Train Data
    train(data, run)


if __name__ == '__main__':
    main()

