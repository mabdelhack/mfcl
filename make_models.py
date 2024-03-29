from models.modulate_layer_network import CompensateDnn
from xgboost import XGBClassifier

def get_models(model_name, preprocessing, layer_size, compensation_network, deep_network=2):
    model_list = list()
    imputation_method = list()
    if model_name == 'mfcl':
        comp_layers = compensation_network
        model = CompensateDnn(input_dimensions=[len(preprocessing['numerical_inputs']),
                                                0],
                              hidden_layers_dimensions=[layer_size, deep_network],
                              activation_function_choice='relu',
                              number_of_outputs=1,
                              compensation_layer_sizes=comp_layers,
                              compensation_activation='relu',
                              compensation_dropout=0.0,
                              compensation_layer_location=0,
                              plus=False,
                              nan_input=False,
                              data_input_modulate=True
        )
        model_list.append(model)
        imputation_method.append('default_flag')
    elif model_name == 'xgboost':
        model = XGBClassifier(n_estimators=100, max_depth=6)
        model_list.append(model)
        imputation_method.append('none')

    return model_list, imputation_method
