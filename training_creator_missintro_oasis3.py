from datetime import datetime
from model_train import model_train
from model_train_baseline import model_train as model_train_baseline


time_now = datetime.now()
current_time_str = time_now.strftime("%Y%m%d_%H%M")

model_properties = {'architecture': 'modulate_layer_network_compdrop',
                    'number_of_layers': 2,
                    'size_of_layers': [4, 2],
                    'compensation_layer_location': 0,
                    'activation_function': 'relu',
                    'dropout_rate': 0.5,
                    'compensation_architecture': [8, 4],
                    'compensation_activation': 'relu',
                    'compensation_dropout': 0.0,
                    'plus': False,
                    'nan_input': False,
                    'data_input_modulate': True}

training = dict()
training['name'] = 'oasis3_quantile_highest25'
training['experiment_name'] = training['name'] + '_' + current_time_str
training['description'] = 'This is a model of for predicting dementia from oasis3 data'
training['data_location'] = 'oasis3_quantile_25_20210616_0018'
training['use_gpu'] = False
training['inputs'] = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
training['outputs'] = ['Dementia']
training['number_of_epochs'] = 1000
training['batch_size'] = 64
training['verbose'] = True
training['seed_value'] = 0
training['plotting'] = False

preprocessing = dict()
preprocessing['preop_missing_imputation'] = 'default_flag'
preprocessing['imbalance_compensation'] = 'none'
preprocessing['numerical_inputs'] = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
preprocessing['categorical_inputs'] = []
preprocessing['outputs'] = ['Dementia']
preprocessing['miss_augment'] = None
preprocessing['miss_introduce'] = None
preprocessing['data_subset'] = 1.0

model_parameters = dict()
model_parameters['counter'] = preprocessing['preop_missing_imputation'] + '_' + str(training['seed_value'])
model_parameters['name'] = training['name']
model_parameters['optimizer'] = {'name': 'SGD',
                                 'learning_rate': 0.01,
                                 'momentum': 0.9,
                                 'weight_decay': 8e-5}
model_parameters['cost_function'] = 'cross_entropy'
model_parameters['model_construction'] = model_properties
model_parameters['loss_weights'] = [1.0, 1.0]
model_parameters['time_series_length'] = []
model_parameters['lr_scheduler'] = None
if model_properties['architecture'] == 'xgboost':
    model_train_baseline(model_parameters, training, preprocessing)
else:
    model_train(model_parameters, training, preprocessing)
