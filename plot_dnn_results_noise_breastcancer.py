import os
from glob import glob
from random import seed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import roc_auc_score, average_precision_score

n_bootstraps = 1000

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    seed(worker_seed)

target_name = 'breastcancer'
testing_paradigm = 'noise'
max_epoch = 49
results_location = ''  # add the results folder name
result_data = './results/{}/'.format(results_location)
model_list = glob('{}*/'.format(result_data))

max_epochs = list()
model_location = list()
models_df = pd.DataFrame()

for model_folder in model_list:
    outdata = pd.read_csv(
        os.path.join(model_folder, 'crossvalidation_split_0', 'epoch_{:0>3}_prediction.csv'.format(max_epoch)))
    outputs = outdata['prediction']
    targets = outdata['target']

    auroc_folds = []
    auprc_folds = []
    rng = np.random.RandomState(42)
    for i in range(n_bootstraps):
        indices = rng.randint(0, len(outputs), len(outputs))
        if len(np.unique(outputs[indices])) < 2:
            continue
        roc_sc = roc_auc_score(targets[indices], outputs[indices])
        prc_sc = average_precision_score(targets[indices], outputs[indices])
        auroc_folds.append(roc_sc)
        auprc_folds.append(prc_sc)
    auroc_folds = np.array(auroc_folds)
    auprc_folds = np.array(auprc_folds)
    bootstrap_fold = pd.DataFrame({'model': [model_folder] * 1000,
                                   'bootstrap_fold': np.arange(1000),
                                   'auroc': auroc_folds,
                                   'auprc': auprc_folds})
    models_df = pd.concat([models_df, bootstrap_fold], ignore_index=True)
    auroc_folds.sort()
    auprc_folds.sort()

models_df.to_csv(('./stats/bootstrap_{}_{}.csv'.format(target_name, testing_paradigm)))

# Plot results
fig, axs = plt.subplots(1, 2)
model_list_plot = model_list
sns.barplot(data=models_df, x='model', y='auprc', ax=axs[0])
sns.barplot(data=models_df, x='model', y='auroc', ax=axs[1])
plt.show()