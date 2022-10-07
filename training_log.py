import json
import os.path
from datetime import datetime
from os import mkdir
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, precision_recall_curve
from torch import save as torchsave


class ResultLogger(object):
    def __init__(self, model_parameters, training, preprocessing, experiment_name=None):
        self.metadata = dict()
        self.metadata['model_parameters'] = model_parameters
        self.metadata['training'] = training
        self.metadata['preprocessing'] = preprocessing

        self.ID = np.array([]).reshape(0, 1)
        self.outputs = np.array([]).reshape(0, 1)
        self.targets = np.array([]).reshape(0, 1)
        self.statistics = dict()
        time_now = datetime.now()
        current_time_str = time_now.strftime("%Y%m%d_%H%M%S")
        if (experiment_name is not None) and (not os.path.isdir('./results/{}'.format(experiment_name))):
            try:
                mkdir('./results/{}'.format(experiment_name))
            except OSError:
                pass
        if not ('counter' in model_parameters.keys()):
            model_parameters['counter'] = 0
        self.directory = './results/{}/{}_{}_{}_{}'.format(experiment_name,
                                                        model_parameters['name'],
                                                        model_parameters['model_construction']['architecture'],
                                                        model_parameters['counter'],
                                                        current_time_str)
        self.base_directory = self.directory
        Path(self.directory).mkdir(parents=True, exist_ok=True)
        # if not os.path.isdir(self.directory):
        #     mkdir(self.directory)

    def add_results(self, ID, output, target, torch=True, number_of_classes=1):
        if torch:
            self.ID = np.vstack([self.ID, ID.detach().numpy().reshape((-1, 1))])
            self.outputs = np.vstack([self.outputs, output.detach().numpy().reshape((-1, number_of_classes))])
            self.targets = np.vstack([self.targets, target.detach().numpy().reshape((-1, 1))])
        else:
            self.ID = np.vstack([self.ID, np.array(ID).reshape((-1, 1))])
            self.outputs = np.vstack([self.outputs, output.reshape((-1, number_of_classes))])
            self.targets = np.vstack([self.targets, target.reshape((-1, 1))])

    def reset_results(self, number_of_classes=1):
        self.ID = np.array([]).reshape(0, 1)
        self.outputs = np.array([]).reshape(0, number_of_classes)
        self.targets = np.array([]).reshape(0, 1)
        self.statistics = dict()

    def compute_auroc(self):
        auroc = roc_auc_score(self.targets, self.outputs)
        return auroc

    def compute_statistics(self, save=False, epoch=None, multiclass=False):
        if not multiclass:
            precision, recall, _ = precision_recall_curve(self.targets, self.outputs)
            average_precision = average_precision_score(self.targets, self.outputs)
            false_positive_rate, true_positive_rate, thresholds = roc_curve(self.targets, self.outputs)
        else:
            self.outputs = np.exp(self.outputs)
        auroc = roc_auc_score(self.targets, self.outputs, multi_class='ovr')
        if not multiclass:
            self.statistics['precision'] = precision
            self.statistics['recall'] = recall
            self.statistics['auprc'] = average_precision
            self.statistics['fpr'] = false_positive_rate
            self.statistics['tpr'] = true_positive_rate
            self.statistics['roc_thresholds'] = thresholds
        self.statistics['auroc'] = auroc
        if save:
            prediction_filename = 'epoch_{0:03d}_prediction.csv'.format(epoch)
            statistics_filename = 'epoch_{0:03d}_statistics.json'.format(epoch)
            statistics = dict()
            statistics['auroc'] = self.statistics['auroc']
            if not multiclass:
                statistics['auprc'] = self.statistics['auprc']
            with open(os.path.join(self.directory, statistics_filename), 'w') as file_ptr:
                json.dump(statistics, file_ptr)
            if not multiclass:
                predictions = pd.DataFrame({'prediction': self.outputs.squeeze(), 'target': self.targets.squeeze()},
                                           index=self.ID.squeeze())
            else:
                predictions = pd.DataFrame({'prediction': self.outputs.max(axis=1).squeeze(), 'target': self.targets.squeeze()},
                                           index=self.ID.squeeze())
            predictions.to_csv(os.path.join(self.directory, prediction_filename), index_label='ID')

    def plot_roc(self, save=False, epoch=None):
        plt.figure()
        plt.step(self.statistics['fpr'], self.statistics['tpr'], color='darkorange', lw=2, label='ROC curve')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristics')
        plt.legend(loc="lower right")
        if save:
            filename = 'epoch_{0:03d}_roc.pdf'.format(epoch)
            plt.savefig(os.path.join(self.directory, filename))
        plt.close()

    def plot_prc(self, save=False, epoch=None):
        plt.figure()
        plt.step(self.statistics['recall'], self.statistics['precision'], color='darkorange', lw=2, label='PRC curve')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.0])
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision Recall Curve')
        plt.legend(loc="lower right")
        if save:
            filename = 'epoch_{0:03d}_prc.pdf'.format(epoch)
            plt.savefig(os.path.join(self.directory, filename))
        plt.close()

    def compute_loss(self, save=False, epoch=None, verbose=False):
        loss_value = np.mean((self.targets - self.outputs)**2)
        self.statistics['loss'] = loss_value
        if save:
            prediction_filename = 'epoch_{0:03d}_prediction.csv'.format(epoch)
            statistics_filename = 'epoch_{0:03d}_statistics.json'.format(epoch)
            statistics = dict()
            statistics['loss'] = self.statistics['loss']
            with open(os.path.join(self.directory, statistics_filename), 'w') as file_ptr:
                json.dump(statistics, file_ptr)
            predictions = pd.DataFrame({'prediction': self.outputs.squeeze(), 'target': self.targets.squeeze()},
                                       index=self.ID.squeeze())
            predictions.to_csv(os.path.join(self.directory, prediction_filename), index_label='ID')
        if verbose:
            print('Validation loss = {}'.format(self.statistics['loss']))

    def verbose(self, epoch=None):
        print('Epoch {}'.format(epoch))
        if 'auroc' in self.statistics.keys():
            print('AUROC = {}'.format(self.statistics['auroc']))
        if 'auprc' in self.statistics.keys():
            print('AUPRC = {}'.format(self.statistics['auprc']))
        if 'loss' in self.statistics.keys():
            print('Loss = {}'.format(self.statistics['loss']))

    def add_metadata(self, subset, item_name, item_value):
        self.metadata[subset][item_name] = item_value

    def save_metadata(self):
        filename = os.path.join(self.directory, 'metadata.json')
        with open(filename, 'w') as file_ptr:
            json.dump(self.metadata, file_ptr)

    def save_model(self, dict_to_save, epoch):
        filename = 'epoch_{0:03d}_model.pth.tar'.format(epoch)
        torchsave(dict_to_save, os.path.join(self.directory, filename))

    def set_crossvalidation_directory(self, split):
        self.directory = os.path.join(self.base_directory, 'crossvalidation_split_{}'.format(split))
        if not os.path.isdir(self.directory):
            mkdir(self.directory)

    def results_summary(self, number_of_epochs, mode='cross_entropy', plot=True):

        if mode == 'cross_entropy':
            summary_statistics = pd.DataFrame(columns=['auroc', 'auprc'])
            for epoch in range(number_of_epochs):
                statistics_filename = 'epoch_{0:03d}_statistics.json'.format(epoch)
                with open(os.path.join(self.directory, statistics_filename), 'r') as file_ptr:
                    data = json.load(file_ptr)
                    summary_statistics = summary_statistics.append(data, ignore_index=True)
            summary_statistics.to_csv(os.path.join(self.directory, 'summary_statistics.csv'))
            if plot:
                plt.figure()
                plt.plot(np.arange(number_of_epochs), summary_statistics['auroc'].values, color='black', lw=1, label='AUROC')
                plt.plot(np.arange(number_of_epochs), summary_statistics['auprc'].values, color='grey', lw=1, label='AUPRC')
                plt.ylim([0.0, 1.2])
                plt.xlim([0.0, number_of_epochs])
                plt.xlabel('Epoch')
                plt.ylabel('Area')
                plt.legend(loc="upper right")
                filename = 'Area_under_ROC_PRC_curves_summary.pdf'

        elif mode == 'mean_squared_error':

            summary_statistics = pd.DataFrame(columns=['loss'])
            for epoch in range(number_of_epochs):
                statistics_filename = 'epoch_{0:03d}_statistics.json'.format(epoch)
                with open(os.path.join(self.directory, statistics_filename), 'r') as file_ptr:
                    data = json.load(file_ptr)
                    summary_statistics = summary_statistics.append(data, ignore_index=True)
            summary_statistics.to_csv(os.path.join(self.directory, 'summary_statistics.csv'))
            if plot:
                plt.figure()
                plt.plot(np.arange(number_of_epochs), summary_statistics['loss'].values, color='black', lw=1,
                         label='Loss')
                plt.xlim([0.0, number_of_epochs])
                plt.xlabel('Epoch')
                plt.ylabel('Loss')
                plt.legend(loc="upper right")
                filename = 'Loss_summary.pdf'

        elif mode == 'categorical_crossentropy':
            summary_statistics = pd.DataFrame(columns=['auroc'])
            for epoch in range(number_of_epochs):
                statistics_filename = 'epoch_{0:03d}_statistics.json'.format(epoch)
                with open(os.path.join(self.directory, statistics_filename), 'r') as file_ptr:
                    data = json.load(file_ptr)
                    summary_statistics = summary_statistics.append(data, ignore_index=True)
            summary_statistics.to_csv(os.path.join(self.directory, 'summary_statistics.csv'))

        if (mode is not 'categorical_crossentropy') and plot:
            plt.savefig(os.path.join(self.directory, filename))
            plt.close()

    def aggregate_results(self, number_of_crossvalidation, mode='cross_entropy'):
        result_columns = ['split_{}'.format(x) for x in range(number_of_crossvalidation)]
        if mode == 'cross_entropy':
            auroc = pd.DataFrame(columns=result_columns)
            auprc = pd.DataFrame(columns=result_columns)
            for split in range(number_of_crossvalidation):
                split_result_file = os.path.join(self.base_directory,
                                                 'crossvalidation_split_{}'.format(split),
                                                 'summary_statistics.csv')
                split_results = pd.read_csv(split_result_file)
                auroc['split_{}'.format(split)] = split_results['auroc']
                auprc['split_{}'.format(split)] = split_results['auprc']

            auroc.to_csv(os.path.join(self.base_directory, 'auroc_aggregate.csv'))
            auprc.to_csv(os.path.join(self.base_directory, 'auprc_aggregate.csv'))
        elif mode == 'mean_squared_error':
            loss = pd.DataFrame(columns=result_columns)
            for split in range(number_of_crossvalidation):
                split_result_file = os.path.join(self.base_directory,
                                                 'crossvalidation_split_{}'.format(split),
                                                 'summary_statistics.csv')
                split_results = pd.read_csv(split_result_file)
                loss['split_{}'.format(split)] = split_results['loss']

            loss.to_csv(os.path.join(self.base_directory, 'loss_aggregate.csv'))
        if mode == 'categorical_crossentropy':
            auroc = pd.DataFrame(columns=result_columns)
            for split in range(number_of_crossvalidation):
                split_result_file = os.path.join(self.base_directory,
                                                 'crossvalidation_split_{}'.format(split),
                                                 'summary_statistics.csv')
                split_results = pd.read_csv(split_result_file)
                auroc['split_{}'.format(split)] = split_results['auroc']

            auroc.to_csv(os.path.join(self.base_directory, 'auroc_aggregate.csv'))


