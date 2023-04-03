from glob import glob
import numpy as np
import pandas as pd
import torch
from torch.utils.data.dataset import Dataset

from EMRData.EMRData import EMRData


class EmrDataLoader(Dataset):
    def __init__(self, train_val_location=None, input_variables=None, output_variables=None, torch_tensor=True,
                 collate=None, num_of_gpus=1, load_missing_flag=False):
        dataset_file_location = glob('./data/{}/training_validation_dataset.*'.format(train_val_location))[0]
        metadata_file_location = './data/{}/metadata.json'.format(train_val_location)
        self.data_store = EMRData(source='presaved', file_path=dataset_file_location, metadata=metadata_file_location)
        self.number_of_splits = self.data_store.metadata['number_of_splits']
        self.selection_column = None
        self.split_number = None
        self.training_data = None
        self.validation_data = None
        self.numerical_variables = None
        self.categorical_variables = None
        self.input_variables = input_variables
        self.output_variables = output_variables
        self.mode = 'train'
        self.torch = torch_tensor
        self.categorization_done = False
        self.numerical_column_selector = []
        self.categorical_column_selector = []
        self.output_column_selector = []
        self.nan_flag = None
        self.training_flag = None
        self.validation_flag = None
        self.input_column_selector = None
        self.collate = collate
        self.num_of_gpus = num_of_gpus
        self.original_validation = None
        self.removed_data_flag = None
        self.original_training = None
        self.removed_training_flag = None
        self.train_val_location = train_val_location
        if load_missing_flag:
            flag_file_location = './data/{}/nan_flag.csv'.format(train_val_location)
            self.nan_flag = pd.read_csv(flag_file_location, index_col=0)
            for output_variable in output_variables:
                self.nan_flag[output_variable] = np.nan

    def preprocess(self, preprocessing, split_number):

        self.split_number = split_number
        self.numerical_variables = preprocessing['numerical_inputs']
        self.selection_column = 'train_split_{}'.format(self.split_number)
        self.data_store.normalized_data = pd.DataFrame()


        if preprocessing['preop_missing_imputation'] == 'flag':
            self.nan_flag = self.data_store.flag_missing(preprocessing['numerical_inputs']
                                                         + preprocessing['categorical_inputs'])
            nan_mode = 'flag'
        elif preprocessing['preop_missing_imputation'] == 'default_flag':
            self.nan_flag = self.data_store.flag_missing(preprocessing['numerical_inputs']
                                                         + preprocessing['categorical_inputs'])
            self.data_store.default_impute(variable_list=preprocessing['numerical_inputs'])
            nan_mode = 'flag'
        elif preprocessing['preop_missing_imputation'] == 'default':
            self.data_store.default_impute(variable_list=preprocessing['numerical_inputs'])
            nan_mode = 'include'
        elif preprocessing['preop_missing_imputation'] == 'mean':
            self.data_store.mean_impute(variable_list=preprocessing['numerical_inputs'])
            nan_mode = 'include'

        else:
            self.nan_flag = self.data_store.flag_missing_keep_nan(preprocessing['numerical_inputs']
                                                         + preprocessing['categorical_inputs'])


            nan_mode = None

        self.data_store.normalize(variable_list=preprocessing['numerical_inputs'],
                                  normalizing_values=
                                  self.data_store.metadata['mean_std_normalization'][self.split_number],
                                  apply_normalization=True,
                                  nan_flag=self.nan_flag)

        self.categorical_variables = preprocessing['categorical_inputs']
        self.categorical_variables, self.nan_flag = self.data_store.categorical_encoding(
            variable_list=preprocessing['categorical_inputs'],
            nan_mode=nan_mode,
            nan_flag=self.nan_flag)
        if 'numerical_outputs' in preprocessing.keys():
            normalizing_value = self.data_store.metadata['mean_std_normalization'] \
                [self.split_number][preprocessing['numerical_outputs'][0]]
            normalized_output = (self.data_store.data.iloc[self.output_variables] - normalizing_value[0]) \
                                / normalizing_value[1]
            self.data_store.data[self.output_variables] = normalized_output

        self.data_store.normalized_data = pd.concat([self.data_store.normalized_data,
                                                     self.data_store.data[self.output_variables]],
                                                    axis=1)

        if self.nan_flag is not None and not self.nan_flag.empty:
            self.data_store.normalized_data, self.nan_flag = self.data_store.normalized_data.align(self.nan_flag,
                                                                                                   join='left',
                                                                                                   axis=None)

        self.training_data = self.data_store.normalized_data.loc[self.data_store.data[self.selection_column]]
        self.validation_data = self.data_store.normalized_data.loc[~self.data_store.data[self.selection_column]]

        if type(self.data_store.normalized_data.index) == pd.core.indexes.multi.MultiIndex:
            index_train = self.training_data.index.get_level_values(0).drop_duplicates()
            index_train = index_train[:np.int(np.floor(len(index_train) / self.num_of_gpus) *
                                              self.num_of_gpus)]
            index_valid = self.validation_data.index.get_level_values(0).drop_duplicates()
            index_valid = index_valid[:np.int(np.floor(len(index_valid) / self.num_of_gpus) *
                                              self.num_of_gpus)]

            self.training_data = self.training_data.loc[index_train]
            self.validation_data = self.validation_data.loc[index_valid]
        if self.nan_flag is not None and not self.nan_flag.empty:
            self.training_flag = self.nan_flag.loc[self.data_store.data[self.selection_column]]
            self.validation_flag = self.nan_flag.loc[~self.data_store.data[self.selection_column]]
        if 'miss_augment' in preprocessing.keys() and preprocessing['miss_augment'] is not None:
            self.miss_augmentation(preprocessing['miss_augment'])

        if 'miss_introduce' in preprocessing.keys() and preprocessing['miss_introduce'] is not None:
            if 'compare_removed' in preprocessing.keys():
                self.validation_data, self.validation_flag,\
                self.original_validation, self.removed_data_flag = \
                    self.miss_introduction(preprocessing['miss_introduce'],
                                           self.validation_data,
                                           self.validation_flag,
                                           self.input_variables,
                                           output_variables=self.output_variables,
                                           return_original=preprocessing['compare_removed'])
            else:
                self.validation_data, self.validation_flag = \
                    self.miss_introduction(preprocessing['miss_introduce'],
                                           self.validation_data,
                                           self.validation_flag,
                                           self.input_variables,
                                           output_variables=self.output_variables)

        if 'miss_introduce_quantile' in preprocessing.keys() and preprocessing['miss_introduce_quantile'] is not None:
            if 'compare_removed' in preprocessing.keys():
                self.original_validation, self.removed_data_flag = \
                    self.miss_introduction_quantile(preprocessing['miss_introduce_quantile'],
                                                    return_original=preprocessing['compare_removed'])
            else:
                self.miss_introduction_quantile(preprocessing['miss_introduce_quantile'])

        if 'miss_introduce_feature' in preprocessing.keys() and preprocessing['miss_introduce_feature'] is not None:
            if 'compare_removed' in preprocessing.keys():
                self.original_validation, self.removed_data_flag = \
                    self.miss_introduction_feature(preprocessing['miss_introduce_feature'],
                                                    return_original=preprocessing['compare_removed'])
            else:
                self.miss_introduction_feature(preprocessing['miss_introduce_feature'])


        self.input_column_selector = [True if ((column in self.numerical_variables) or
                                               (column in self.categorical_variables)) else False
                                      for column in self.training_data.columns]
        self.numerical_column_selector = [True if column in self.numerical_variables else False
                                          for column in self.training_data.columns]
        self.categorical_column_selector = [True if column in self.categorical_variables else False
                                            for column in self.training_data.columns]
        self.output_column_selector = [True if column in self.output_variables else False
                                       for column in self.training_data.columns]

    def set_mode(self, mode):
        self.mode = mode

    def __getitem__(self, item):
        if type(self.training_data.index) == pd.core.indexes.multi.MultiIndex:
            if self.mode == 'train':
                ID = self.training_data.index.get_level_values(0).drop_duplicates()[item]
                x_numeric = self.training_data.loc[ID, self.numerical_column_selector].values
                x_categorical = self.training_data.loc[ID, self.categorical_column_selector].values
                if not self.nan_flag.empty:
                    x_flags = self.training_flag.loc[ID, self.input_column_selector].values
                y = self.training_data.loc[ID, self.output_column_selector].values
                y = y[0]
            elif self.mode == 'validation':
                ID = self.validation_data.index.get_level_values(0).drop_duplicates()[item]
                x_numeric = self.validation_data.loc[ID, self.numerical_variables].values
                x_categorical = self.validation_data.loc[ID, self.categorical_variables].values
                if not self.nan_flag.empty:
                    x_flags = self.validation_flag.loc[ID, self.input_column_selector].values
                y = self.validation_data.loc[ID, self.output_variables].values
                y = y[0]
        else:
            if self.mode == 'train':
                ID = self.training_data.index[item]
                x_numeric = self.training_data.iloc[item, self.numerical_column_selector].values
                x_categorical = self.training_data.iloc[item, self.categorical_column_selector].values
                if self.original_validation is not None:
                    original_training = self.original_training.iloc[item, self.input_column_selector].values
                if not self.nan_flag.empty:
                    x_flags = self.training_flag.iloc[item, self.input_column_selector].values
                    if self.removed_training_flag is not None:
                        removed_training_flag = self.removed_training_flag.iloc[item].values
                y = self.training_data.iloc[item, self.output_column_selector].values
            elif self.mode == 'validation':
                ID = self.validation_data.index[item]
                x_numeric = self.validation_data.loc[ID, self.numerical_variables].values
                x_categorical = self.validation_data.loc[ID, self.categorical_variables].values
                if self.original_validation is not None:
                    original_validation = self.original_validation.loc[ID, self.input_variables].values
                if not self.nan_flag.empty:
                    x_flags = self.validation_flag.iloc[item, self.input_column_selector].values
                    if self.removed_data_flag is not None:
                        removed_data_flag = self.removed_data_flag.iloc[item].values
                y = self.validation_data.loc[ID, self.output_variables].values
        x_numeric = x_numeric.astype(float)
        x_categorical = x_categorical.astype(float)
        if not self.nan_flag.empty:
            x_flags = x_flags.astype(float)
        y = y.astype(float)
        if self.torch:
            x_numeric = torch.from_numpy(x_numeric).float()
            x_categorical = torch.from_numpy(x_categorical).float()
            if not self.nan_flag.empty:
                x_flags = torch.from_numpy(x_flags).float()
            y = torch.from_numpy(y).float()

        if not self.nan_flag.empty:
            x = [x_numeric, x_categorical, x_flags]
        else:
            x = [x_numeric, x_categorical]

        if self.mode == 'validation' and self.removed_data_flag is not None:
            x.append(torch.from_numpy(original_validation).float())
            x.append(torch.from_numpy(removed_data_flag).float())

        if self.mode == 'train' and self.removed_data_flag is not None:
            x.append(torch.from_numpy(original_training).float())
            x.append(torch.from_numpy(removed_training_flag).float())

        if self.collate is not None:
            x = torch.cat(x, dim=1)

        return ID, x, y

    def __len__(self):
        if self.training_data is not None:
            if type(self.training_data.index) == pd.core.indexes.multi.MultiIndex:
                if self.mode == 'train':
                    return len(self.training_data.index.get_level_values(0).drop_duplicates())
                elif self.mode == 'validation':
                    return len(self.validation_data.index.get_level_values(0).drop_duplicates())
            else:
                if self.mode == 'train':
                    return self.training_data.shape[0]
                elif self.mode == 'validation':
                    return self.validation_data.shape[0]
        else:
            if self.mode == 'train':
                return self.data_store.data['train_split_0'].sum()
            elif self.mode == 'validation':
                return (~self.data_store.data['train_split_0']).sum()

    def miss_introduction(self, missing_proportion, input_data, input_flag, input_variables,
                          output_variables=None, return_original=False):
        from simulated_data_generator import SimulatedDataGenerator
        temp_data = input_data
        temp_flag = input_flag

        data, flag = SimulatedDataGenerator.generate_missing(temp_data[input_variables],
                                                             missing_proportion, fill_na=0.0)
        flag = np.logical_and(np.logical_not(temp_flag[input_variables].values), flag)
        removed_flag = pd.DataFrame(data=flag, columns=input_variables).astype(bool)
        removed_flag.set_index(temp_flag.index, inplace=True)

        flag = np.logical_or(temp_flag[input_variables].values, flag)
        input_flag = pd.DataFrame(data=flag, columns=input_variables).astype(bool)
        input_flag.set_index(temp_flag.index, inplace=True)
        input_flag[output_variables] = temp_flag[output_variables]
        input_data = pd.DataFrame(data=data, columns=input_variables)
        input_data.set_index(temp_data.index, inplace=True)
        input_data[output_variables] = temp_data[output_variables]

        empty_samples = flag.sum(axis=1) == flag.shape[1]
        input_data = input_data.loc[~empty_samples, :]
        input_flag = input_flag.loc[~empty_samples, :]
        if return_original:
            return input_data, input_flag, temp_data, removed_flag
        else:
            return input_data, input_flag

    def miss_introduction_quantile(self, quantile, mode='bigger', return_original=False):
        temp_data = self.validation_data
        temp_flag = self.validation_flag

        if mode == 'bigger':
            quantile_flag = temp_data[self.input_variables] > temp_data[self.input_variables].quantile(quantile)
        elif mode == 'smaller':
            quantile_flag = temp_data[self.input_variables] < temp_data[self.input_variables].quantile(quantile)
        self.validation_data[quantile_flag] = 0.0

        self.validation_flag[self.input_variables] = temp_flag[self.input_variables] | quantile_flag

        empty_samples = self.validation_flag.sum(axis=1) == self.validation_flag.shape[1]
        self.validation_data = self.validation_data.loc[~empty_samples, :]
        self.validation_flag = self.validation_flag.loc[~empty_samples, :]
        if return_original:
            return temp_data, quantile_flag

    def miss_introduction_feature(self, number_of_features, return_original=False):
        temp_data = self.validation_data
        temp_flag = self.validation_flag

        quantile_flag = self.validation_flag
        quantile_flag.iloc[:, :number_of_features] = True #remove the first features (not randomly chosen)
        quantile_flag = quantile_flag[self.input_variables]
        self.validation_data[quantile_flag] = 0.0

        self.validation_flag[self.input_variables] = temp_flag[self.input_variables] | quantile_flag

        empty_samples = self.validation_flag.sum(axis=1) == self.validation_flag.shape[1]
        self.validation_data = self.validation_data.loc[~empty_samples, :]
        self.validation_flag = self.validation_flag.loc[~empty_samples, :]
        if return_original:
            return temp_data, quantile_flag
