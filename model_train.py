from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from training_log import ResultLogger
import numpy as np
from random import seed
import os
from emr_data_loader import EmrDataLoader


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    seed(worker_seed)


def model_train(model_parameters, training, preprocessing):
    if 'seed_value' in training.keys():
        seed_value = training['seed_value']
    else:
        seed_value = 42
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if 'number_of_classes' not in model_parameters:
        model_parameters['number_of_classes'] = 1
    if torch.cuda.is_available():
        use_gpu = training['use_gpu']
        if use_gpu is None:
            use_gpu = False
    else:
        use_gpu = False
    model_construction = model_parameters['model_construction']
    experiment_name = training['experiment_name']

    optimizer_construction = model_parameters['optimizer']
    if 'lr_scheduler' in model_parameters.keys():
        lr_scheduler_construction = model_parameters['lr_scheduler']
    if 'load_nan' in preprocessing and preprocessing['load_nan']:
        load_nan = True
    else:
        load_nan = False
    # Put Data Loader here
    data_loader = EmrDataLoader(train_val_location=training['data_location'],
                                input_variables=training['inputs'],
                                output_variables=training['outputs'],
                                torch_tensor=True,
                                load_missing_flag=load_nan)

    train_loader = DataLoader(data_loader, batch_size=training['batch_size'],
                              shuffle=True, num_workers=0, worker_init_fn=seed_worker)

    result_logger = ResultLogger(model_parameters, training, preprocessing, experiment_name)

    for crossval_split in range(data_loader.number_of_splits):
        if training['verbose']:
            print('Cross-validation split number {}'.format(crossval_split))
        result_logger.set_crossvalidation_directory(crossval_split)
        data_loader.preprocess(preprocessing=preprocessing, split_number=crossval_split)

        if model_construction['architecture'] == 'modulate_layer_network_compdrop':
            from models.modulate_layer_network_compdropout import CompensateDnn

            model = CompensateDnn(input_dimensions=[len(preprocessing['numerical_inputs']),
                                                    len(data_loader.categorical_variables)],
                                  hidden_layers_dimensions=model_construction['size_of_layers'],
                                  activation_function_choice=model_construction['activation_function'],
                                  number_of_outputs=len(training['outputs']),
                                  compensation_layer_sizes=model_construction['compensation_architecture'],
                                  compensation_activation=model_construction['compensation_activation'],
                                  compensation_dropout=model_construction['compensation_dropout'],
                                  compensation_layer_location=model_construction['compensation_layer_location'],
                                  number_of_classes=model_parameters['number_of_classes'],
                                  plus=model_construction['plus'],
                                  nan_input=model_construction['nan_input'],
                                  data_input_modulate=model_construction['data_input_modulate'])

        if use_gpu:
            model = model.cuda()
        if optimizer_construction['name'] == 'SGD':
            optimizer = optim.SGD(model.parameters(),
                                  lr=optimizer_construction['learning_rate'],
                                  momentum=optimizer_construction['momentum'],
                                  weight_decay=optimizer_construction['weight_decay'])

        elif optimizer_construction['name'] == 'Adam':
            optimizer = optim.Adam(model.parameters(),
                                   lr=optimizer_construction['learning_rate'],
                                   betas=optimizer_construction['betas'],
                                   amsgrad=optimizer_construction['amsgrad'],
                                   eps=optimizer_construction['eps'],
                                   weight_decay=optimizer_construction['weight_decay'])

        if ('lr_scheduler' in model_parameters.keys()) and (model_parameters['lr_scheduler'] is not None):
            lr_scheduler_construction = model_parameters['lr_scheduler']
            if lr_scheduler_construction['name'] == 'LambdaLR':
                lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer,
                                                            lr_scheduler_construction['lr_lambda'],
                                                            last_epoch=lr_scheduler_construction['last_epoch'])
            elif lr_scheduler_construction['name'] == 'MultiplicativeLR':
                lr_scheduler = \
                    optim.lr_scheduler.MultiplicativeLR(optimizer,
                                                        lr_scheduler_construction['lr_lambda'],
                                                        last_epoch=lr_scheduler_construction['last_epoch'])
            elif lr_scheduler_construction['name'] == 'StepLR':
                lr_scheduler = optim.lr_scheduler.StepLR(optimizer,
                                                         lr_scheduler_construction['step_size'],
                                                         gamma=lr_scheduler_construction['gamma'],
                                                         last_epoch=lr_scheduler_construction['last_epoch'])
            elif lr_scheduler_construction['name'] == 'MultiStepLR':
                lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer,
                                                         lr_scheduler_construction['milestone'],
                                                         gamma=lr_scheduler_construction['gamma'],
                                                         last_epoch=lr_scheduler_construction['last_epoch'])
            elif lr_scheduler_construction['name'] == 'ExponentialLR':
                lr_scheduler = \
                    optim.lr_scheduler.ExponentialLR(optimizer,
                                                     lr_scheduler_construction['gamma'],
                                                     last_epoch=lr_scheduler_construction['last_epoch'])
            elif lr_scheduler_construction['name'] == 'CosineAnnealingLR':
                lr_scheduler = \
                    optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                         lr_scheduler_construction['T_max'],
                                                         eta_min=lr_scheduler_construction['eta_min'],
                                                         last_epoch=lr_scheduler_construction['last_epoch'])
            elif lr_scheduler_construction['name'] == 'ReduceLROnPlateau':
                lr_scheduler = \
                    optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                         mode=lr_scheduler_construction['mode'],
                                                         factor=lr_scheduler_construction['factor'],
                                                         patience=lr_scheduler_construction['patience'],
                                                         verbose=lr_scheduler_construction['verbose'],
                                                         threshold=lr_scheduler_construction['threshold'],
                                                         threshold_mode=lr_scheduler_construction['threshold_mode'],
                                                         cooldown=lr_scheduler_construction['cooldown'],
                                                         min_lr=lr_scheduler_construction['min_lr'],
                                                         eps=lr_scheduler_construction['eps'])
        if model_parameters['cost_function'] == 'mean_squared_error':
            loss_function = torch.nn.MSELoss()
        elif model_parameters['cost_function'] == 'categorical_crossentropy':
            loss_function = torch.nn.NLLLoss()

        clipping_value = 0.1  # clip grad to prevent nan loss
        torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)

        result_logger.save_metadata()
        best_metric = np.Inf if model_parameters['cost_function'] == 'mean_squared_error' else 0
        no_improvement_counter = 0
        for epoch in range(training['number_of_epochs']):
            if training['verbose']:
                print('Epoch {}'.format(epoch))
            data_loader.set_mode('train')
            if training['verbose']:
                progress_bar = tqdm(enumerate(iter(train_loader)),
                                    total=np.ceil(len(data_loader)/training['batch_size']).astype(int))
            else:
                progress_bar = enumerate(iter(train_loader))
            for batch_idx, (IDs, x_train, y_train) in progress_bar:
                optimizer.zero_grad()

                if use_gpu:
                    if type(x_train) == list:
                        x_train = [x.cuda() for x in x_train]
                    else:
                        x_train = x_train.cuda()
                    y_train = y_train.cuda()
                y_hat = model(x_train)
                if model_parameters['cost_function'] in ['categorical_crossentropy']:
                    # loss_element = list()
                    # for output_idx, output_target in enumerate(training['outputs']):
                    loss = loss_function(y_hat, y_train.squeeze().type(torch.LongTensor))
                    # loss = sum(loss_element)
                elif model_parameters['cost_function'] in ['mean_squared_error']:
                    # loss_element = list()
                    # for output_idx, output_target in enumerate(training['outputs']):
                    loss = loss_function(y_hat, y_train.squeeze().type(torch.FloatTensor))
                    # loss = sum(loss_element)
                else:
                    loss = weighted_binary_logloss(y_hat, y_train, weights=model_parameters['loss_weights'])
                loss.backward()
                optimizer.step()
                if training['verbose']:
                    progress_bar.set_postfix_str(s='Loss = {0:0.5f}'.format(loss.item()))

            if training['verbose']:
                print('Validation set testing...')
            data_loader.set_mode('validation')
            result_logger.reset_results(number_of_classes=model_parameters['number_of_classes'])
            for batch_idx, (IDs, x_validation, y_validation) in enumerate(iter(train_loader)):
                if use_gpu:
                    if type(x_validation) == list:
                        x_validation = [x.cuda() for x in x_validation]
                    else:
                        x_validation = x_validation.cuda()
                    y_validation = y_validation.cuda()

                model.eval()
                y_hat = model(x_validation)
                if use_gpu:
                    y_hat = y_hat.cpu()
                    y_validation = y_validation.cpu()
                result_logger.add_results(IDs, y_hat, y_validation, number_of_classes=model_parameters['number_of_classes'])
            if model_parameters['cost_function'] == 'mean_squared_error':
                result_logger.compute_loss(save=True, epoch=epoch, verbose=training['verbose'])
            else:
                result_logger.compute_statistics(save=True, epoch=epoch,
                                                 multiclass=model_parameters['number_of_classes'] > 1)
                if training['verbose']:
                    result_logger.verbose(epoch=epoch)
                if (model_parameters['number_of_classes'] < 2) and (training['plotting']):
                    result_logger.plot_roc(save=True, epoch=epoch)
                    result_logger.plot_prc(save=True, epoch=epoch)
            model_state = {'model_state': model.state_dict(),
                           'optimizer_state': optimizer.state_dict(),
                           'epoch': epoch}
            result_logger.save_model(model_state, epoch)
            if 'lr_scheduler' in model_parameters.keys() and (model_parameters['lr_scheduler'] is not None):
                if 'target_variable' in lr_scheduler_construction.keys():
                    if lr_scheduler_construction['target_variable'] == 'auroc':
                        validation_metric = result_logger.compute_auroc()
                        best_metric = np.maximum(best_metric, validation_metric)
                    elif lr_scheduler_construction['target_variable'] == 'loss':
                        if model_parameters['cost_function'] == 'mean_squared_error':
                            validation_metric = loss_function(torch.from_numpy(result_logger.outputs).float(),
                                                              torch.from_numpy(result_logger.targets).float())
                        else:
                            validation_metric = weighted_binary_logloss(torch.from_numpy(result_logger.outputs).float(),
                                                                        torch.from_numpy(result_logger.targets).float())
                        validation_metric = validation_metric.numpy()
                        best_metric = np.minimum(best_metric, validation_metric)
                    lr_scheduler.step(validation_metric)
                else:
                    lr_scheduler.step()
                if best_metric == validation_metric:
                    no_improvement_counter = 0
                else:
                    no_improvement_counter += 1
                    if training['verbose']:
                        print('No improvement for {} epochs'.format(no_improvement_counter))
                if 'early_stopping_threshold' in training.keys():
                    if no_improvement_counter > training['early_stopping_threshold']:
                        break
        result_logger.results_summary(epoch, mode=model_parameters['cost_function'], plot=training['plotting'])
    result_logger.aggregate_results(data_loader.number_of_splits, mode=model_parameters['cost_function'])


def weighted_binary_logloss(x, y, weights=None, reduction_mode='mean', epsilon=1e-12, ignore_nan=True):
    if weights is None:
        weights = [1.0, 1.0]
    loss_vector = -(weights[1] * y * (x + epsilon).log() + weights[0] * (1 - y) * (1 - x + epsilon).log())
    if ignore_nan:
        loss_vector = loss_vector[~torch.isnan(loss_vector)]
    if reduction_mode == 'mean':
        return loss_vector.mean()
    elif reduction_mode == 'sum':
        return loss_vector.sum()
