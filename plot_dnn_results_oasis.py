import torch
import pandas as pd
import os
from emr_data_loader import EmrDataLoader
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score
from random import seed
from make_models import get_models
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from glob import glob
n_bootstraps = 1000

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    seed(worker_seed)

target_name = 'oasis'
max_epoch = 999
results_location = ''  # add the results folder name
testing_paradigm = 'random'  # choose testing paradigm (choices: random, quantile, feature)
result_data = './results/{}/'.format(results_location)
model_list = glob('{}*/'.format(result_data))
auroc_stats_r = pd.DataFrame(columns=model_list, index=[0.0, 0.2, 0.4, 0.6, 0.8])
auprc_stats_r = pd.DataFrame(columns=model_list, index=[0.0, 0.2, 0.4, 0.6, 0.8])


if testing_paradigm == 'random':
    sequence = [0.0, 0.2, 0.4, 0.6, 0.8]
elif testing_paradigm == 'quantile':
    sequence = [0.2, 0.4, 0.6, 0.8]
elif testing_paradigm == 'feature':
    sequence = [1, 2, 3, 4, 5]

max_epochs = list()
model_location = list()
for model_name in model_list:

    max_epochs.append(max_epoch)
    best_auroc_model = os.path.join(model_name,
                                    'crossvalidation_split_0',
                                    'epoch_{:03d}_model.pth.tar'.format(max_epoch))
    model_location.append(best_auroc_model)


data_location = 'oasis3_quantile_25_20210616_0018'
inputs_vars = ['Age', 'EDUC', 'SES', 'MMSE', 'eTIV', 'nWBV', 'ASF']
outputs_vars = ['Dementia']
data_loader = EmrDataLoader(train_val_location=data_location,
                            input_variables=inputs_vars,
                            output_variables=outputs_vars,
                            torch_tensor=True)

train_loader = DataLoader(data_loader, batch_size=128,
                          shuffle=True, num_workers=1)
models_df = pd.DataFrame()

preprocessing = dict()
# Imputation settings
preprocessing['miss_introduce_quantile'] = None
preprocessing['miss_introduce'] = None
preprocessing['miss_introduce_feature'] = None
preprocessing['preop_missing_imputation'] = 'default_flag'

preprocessing['imbalance_compensation'] = 'none'
preprocessing['numerical_inputs'] = inputs_vars
preprocessing['categorical_inputs'] = []
preprocessing['outputs'] = outputs_vars
models, imputation_methods = get_models(preprocessing, 4, [8, 4])

for model_name, model, model_loc, imputation_method in zip(model_list, models, model_location, imputation_methods):
    print(model_name)
    model_param = torch.load(model_loc, map_location=torch.device('cpu'))
    model.load_state_dict(model_param['model_state'])

    seed_value = 0
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed_value)
    seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    for idx, miss_introduce in enumerate(sequence):
        preprocessing['miss_introduce_quantile'] = None
        preprocessing['miss_introduce'] = None
        preprocessing['miss_introduce_feature'] = None

        data_loader = EmrDataLoader(train_val_location=data_location,
                                    input_variables=inputs_vars,
                                    output_variables=outputs_vars,
                                    torch_tensor=True)

        train_loader = DataLoader(data_loader, batch_size=128,
                                  shuffle=True, num_workers=0, worker_init_fn=seed_worker)
        if testing_paradigm == 'random':
            preprocessing['miss_introduce'] = miss_introduce
        elif testing_paradigm == 'quantile':
            preprocessing['miss_introduce_quantile'] = miss_introduce
        elif testing_paradigm == 'feature':
            preprocessing['miss_introduce_feature'] = miss_introduce

        preprocessing['preop_missing_imputation'] = imputation_method
        data_loader.preprocess(preprocessing=preprocessing, split_number=0)
        data_loader.set_mode('validation')
        outputs = np.array([]).reshape(0, 1)
        targets = np.array([]).reshape(0, 1)

        for batch_idx, (IDs, x_validation, y_validation) in enumerate(iter(train_loader)):

            model.eval()
            y_hat = model(x_validation)
            outputs = np.vstack([outputs, y_hat.detach().numpy().reshape((-1, 1))])
            targets = np.vstack([targets, y_validation.detach().numpy().reshape((-1, 1))])

        average_precision = average_precision_score(targets, outputs)
        auroc = roc_auc_score(targets, outputs)
        auroc_stats_r.loc[miss_introduce, model_name] = auroc
        auprc_stats_r.loc[miss_introduce, model_name] = average_precision
        print(miss_introduce, auroc, average_precision)
        auroc_folds = []
        auprc_folds = []
        rng = np.random.RandomState(42)
        for i in range(n_bootstraps):
            indices = rng.randint(0, len(outputs), len(outputs))
            if len(np.unique(targets[indices])) < 2:
                continue
            roc_sc = roc_auc_score(targets[indices], outputs[indices])
            prc_sc = average_precision_score(targets[indices], outputs[indices])
            auroc_folds.append(roc_sc)
            auprc_folds.append(prc_sc)
        auroc_folds = np.array(auroc_folds)
        auprc_folds = np.array(auprc_folds)
        bootstrap_fold = pd.DataFrame({'model':[model_name]*len(auroc_folds),
                                       'miss_value': miss_introduce,
                                       'bootstrap_fold': np.arange(len(auroc_folds)),
                                       'auroc': auroc_folds,
                                       'auprc': auprc_folds})
        models_df = pd.concat([models_df, bootstrap_fold], ignore_index=True)
        auroc_folds.sort()
        auprc_folds.sort()

models_df.to_csv(('./stats/bootstrap_{}_{}.csv'.format(target_name, testing_paradigm)))

# Plot results
fig, axs = plt.subplots(1, 2)
model_list_plot = model_list
sns.lineplot(data=models_df, x='miss_value', y='auprc', ax=axs[0], hue='model')
sns.lineplot(data=models_df, x='miss_value', y='auroc', ax=axs[1], hue='model')
plt.show()