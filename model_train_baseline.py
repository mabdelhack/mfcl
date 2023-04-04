from training_log import ResultLogger
import numpy as np
from emr_data_loader import EmrDataLoader
from sklearn import svm, linear_model
from sklearn import tree
from sklearn import ensemble
from xgboost import XGBClassifier
import pickle
from datetime import datetime
import os.path


def model_train(model_parameters, training, preprocessing):

    model_construction = model_parameters['model_construction']

    experiment_name = training['experiment_name']

    # Put Data Loader here
    data_loader = EmrDataLoader(train_val_location=training['data_location'],
                                input_variables=training['inputs'],
                                output_variables=training['outputs'],
                                torch_tensor=True)

    result_logger = ResultLogger(model_parameters, training, preprocessing, experiment_name)

    for crossval_split in range(data_loader.number_of_splits):
        print('Cross-validation split number {}'.format(crossval_split))
        if 'loss_weights' in model_parameters.keys():
            class_weights = {0: model_parameters['loss_weights'][0], 1: model_parameters['loss_weights'][1]}
        if model_construction['architecture'] == 'random_forests':
            model = ensemble.RandomForestClassifier(n_estimators=model_construction['number_of_estimators'],
                                                    n_jobs=-1,
                                                    class_weight=class_weights)
        elif model_construction['architecture'] == 'svm':
            model = svm.SVC(C=model_construction['C'],
                            kernel=model_construction['kernel'],
                            cache_size=model_construction['cache_size'],
                            verbose=True,
                            probability=True)
        elif model_construction['architecture'] == 'logistic_regression':
            model = linear_model.LogisticRegression(penalty=model_construction['penalty'],
                                                    class_weight=class_weights,
                                                    n_jobs=-1,
                                                    verbose=True)
        elif model_construction['architecture'] == 'decision_trees':
            model = tree.DecisionTreeClassifier(max_depth=model_construction['max_depth'])
        elif model_construction['architecture'] == 'gradient_boosting':
            model = ensemble.GradientBoostingClassifier(n_estimators=model_construction['number_of_estimators'],
                                                        max_depth=model_construction['max_depth'])
        elif model_construction['architecture'] == 'xgboost':
            model = XGBClassifier(max_depth=model_construction['max_depth'], learning_rate=model_construction['learning_rate'], n_estimators=model_construction['number_of_estimators'])

        result_logger.set_crossvalidation_directory(crossval_split)
        data_loader.preprocess(preprocessing=preprocessing, split_number=crossval_split)
        result_logger.add_metadata(subset='preprocessing',
                                   item_name='encoded_categorical_variables',
                                   item_value=data_loader.categorical_variables)
        result_logger.save_metadata()
        data_loader.set_mode('train')
        print('Training {} model'.format(model_construction['architecture']))
        column_selector = list(np.array(data_loader.numerical_column_selector) | np.array(data_loader.categorical_column_selector))
        x_train = data_loader.training_data.iloc[:, column_selector]
        y_train = data_loader.training_data.iloc[:, data_loader.output_column_selector]

        model.fit(x_train.values, y_train.values.ravel())

        print('Validation set testing...')
        data_loader.set_mode('validation')
        result_logger.reset_results()
        IDs = data_loader.validation_data.index
        x_validation = data_loader.validation_data.iloc[:, column_selector]
        y_validation = data_loader.validation_data.iloc[:, data_loader.output_column_selector].values.ravel()
        y_hat = model.predict_proba(x_validation.values)
        y_hat = y_hat[:, 1]
        result_logger.add_results(IDs, y_hat, y_validation, torch=False)
        result_logger.compute_statistics(save=True, epoch=0)
        result_logger.verbose(epoch=0)
        result_logger.plot_roc(save=True, epoch=0)
        result_logger.plot_prc(save=True, epoch=0)
        result_logger.results_summary(1)

        time_now = datetime.now()
        current_time_str = time_now.strftime("%Y%m%d_%H%M")
        filename = '{}_{}.pickle'.format(model_construction['architecture'], current_time_str)
        with open(os.path.join(result_logger.directory, filename), 'wb') as file_ptr:
            pickle.dump(model, file_ptr)

    result_logger.aggregate_results(data_loader.number_of_splits)
