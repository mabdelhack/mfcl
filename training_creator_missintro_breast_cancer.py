from shutil import copyfile
from datetime import datetime
from model_train import model_train
import numpy as np
import sys

# sys.path.append('../email_reminder/')
# from email_notification import send_notification

# Make sure I have a copy of that file with parameters
time_now = datetime.now()
current_time_str = time_now.strftime("%Y%m%d_%H%M")
source = './training_creator.py'
log_destination = './training_creator_log/training_creator_{}.py'.format(current_time_str)
copyfile(source, log_destination)

model_list = list()

model_instance = {'architecture': 'modulate_layer_network_compdrop',
                  'number_of_layers': 2,
                  'size_of_layers': [4, 2],
                  'compensation_layer_location': 0,
                  'activation_function': 'relu',
                  'dropout_rate': 0.5,
                  'compensation_architecture': [4, 4],
                  'compensation_activation': 'relu',
                  'compensation_dropout': 0.0,
                  'plus': False,
                  'nan_input': False,
                  'data_input_modulate': True}
model_list.append(model_instance)

impute_list = ['default_flag']
lr_list = [0.03]

random_seed = range(1)
test_cases = list()
for randseed in random_seed:
    test_cases.append({'random_seed': randseed})
if len(sys.argv) > 1:
    model = model_list[int(sys.argv[1])]
    impute_method = impute_list[int(sys.argv[1])]
    lr = lr_list[int(sys.argv[1])]
else:
    model_idx = 0
    print('Trial mode, running {}th model'.format(model_idx))

    model = model_list[model_idx]
    impute_method = impute_list[model_idx]
    lr = lr_list[model_idx]

print('Number of test cases = {}'.format(len(test_cases)))
for case in test_cases:
    training = dict()
    training['name'] = 'breast_cancer'
    training['experiment_name'] = training['name'] + '_' + current_time_str
    training['description'] = 'This is a model of for predicting breast cancer from sklearn data'
    training['data_location'] = 'sklearn_breastcancer_quantile_highest25removed_20220426_1429'
    training['use_gpu'] = True
    training['inputs'] = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
                          'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
                          'mean fractal dimension']
    training['outputs'] = ['target']
    training['number_of_epochs'] = 50
    training['batch_size'] = 64
    training['verbose'] = True
    training['seed_value'] = case['random_seed']
    training['plotting'] = False


    preprocessing = dict()
    preprocessing['preop_missing_imputation'] = impute_method
    preprocessing['fancy_impute_parameters'] = {'name': 'IterativeImputer', 'parameters': {}}
    preprocessing['auto_impute_parameters'] = {'name': 'mice', 'parameters': {}}
    preprocessing['miwae_impute_parameters'] = {'filename': None}

    preprocessing['missingpy_impute_parameters'] = {'name': 'missforest', 'parameters': {}}
    preprocessing['autoencoder_parameters'] = {'hidden_layers_dimensions': [15],
                                               'activation_function_choice': 'sigmoid',
                                               'dropout_rate': 0.0}
    preprocessing['intraop_missing_imputation'] = []
    preprocessing['imbalance_compensation'] = 'none'
    preprocessing['numerical_inputs'] = ['mean radius', 'mean texture', 'mean perimeter', 'mean area', 'mean smoothness',
                          'mean compactness', 'mean concavity', 'mean concave points', 'mean symmetry',
                          'mean fractal dimension']
    # preprocessing['numerical_outputs'] = []
    preprocessing['categorical_inputs'] = []
    preprocessing['outputs'] = ['target']
    preprocessing['miss_augment'] = None
    # preprocessing['miss_introduce_feature'] = 0.0
    preprocessing['data_subset'] = 1.0

    model_parameters = dict()
    model_parameters['counter'] = impute_method + '_' + str(case['random_seed'])
    model_parameters['name'] = training['name']
    model_parameters['optimizer'] = {'name': 'SGD',
                                     'learning_rate': lr,
                                     'momentum': 0.9,
                                     'weight_decay': 8e-5}
    model_parameters['cost_function'] = 'cross_entropy'
    model_parameters['model_construction'] = model
    model_parameters['loss_weights'] = [1.0, 1.0]
    model_parameters['time_series_length'] = []
    model_parameters['lr_scheduler'] = None
    print(case)
    model_train(model_parameters, training, preprocessing)

