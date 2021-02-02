import argparse
import os

from azureml.core.run import Run

import matplotlib.pyplot as plt
import pandas as pd

from config.config_training import script_params_local

print("defining args")
parser = argparse.ArgumentParser()
parser.add_argument(
    "--remote_execution",
    dest="remote_execution",
    default=False,
)
parser.add_argument(
    "--ds"
    ,dest="dataset"
    ,default=script_params_local['path_local_data']  # path for local debugging
)
parser.add_argument(
    "--out_folder"
    ,dest='out_folder'
    ,default=script_params_local['path_local_data']  # path for local debugging
)

print("parsing args")
args = parser.parse_args()
remote_execution = args.remote_execution
dataset = args.dataset
out_folder = args.out_folder

def preprocessing(data, run):
    # log distinct pregnancy counts
    pregnancies = data.Pregnancies.unique()
    run.log_list('pregnancy categories', pregnancies)
    
    preprocessed_data = data.drop(columns=['PatientID'], axis=1)

    return preprocessed_data


def visualization(data, run):
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


def main(output_folder):
    #  get context from run
    run = Run.get_context()

    #  Load Data
    if remote_execution:
        tab_ds = run.input_datasets['diabetes_dataset']
        df = tab_ds.to_pandas_dataframe()
    else:
        df = pd.read_csv('../data/diabetes.csv', sep=',', decimal='.')

    # Visualization of training data
    visualization(df, run)

    # Preprocess Data
    preprocessed_data = preprocessing(df, run)
    
    # Store the Data
    os.makedirs(output_folder, exist_ok=True)
    output_path = os.path.join(output_folder, 'prepped_data.csv')
    preprocessed_data.to_csv(output_path)

    
if __name__ == '__main__':
    main(output_folder=out_folder)
