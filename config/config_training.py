script_params_remote = [
    '--param_1', 0.8,
    '--remote_execution', True,
    '--path_trained_model', './trained_models/' 
]

script_params_local = {
    'param_1': 0.8
    ,'remote_execution': False
    ,'path_local_data': '../data/diabetes.csv'
    ,'path_trained_model': '../trained_models/'
}
