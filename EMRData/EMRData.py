import json
import os
import warnings
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit

from EMRData.data_variable import DataVariable

"""
=================================
EMRData v0.2 30-Mar-2020
Author: Mohamed Abdelhack
Email: mohamed.abdelhack.37a@kyoto-u.jp
=================================
"""


class EMRData(object):
    io_function_switcher = {
        '.hd5': pd.read_hdf,
        '.pkl': pd.read_pickle,
        '.pickle': pd.read_pickle,
        '.csv': pd.read_csv,
        '.xlsx': pd.read_excel,
        '.xls': pd.read_excel
    }  # functionality for file reading

    def __init__(self, source='presaved', variable_target_file=None, file_path=None, metadata=None, dataset_name=None, max_features=None):
        self.data = pd.DataFrame()
        self.normalized_data = pd.DataFrame()
        self.data_source = source
        self.variable_name = []
        self.data_folder_name = None
        self.test_data = None
        if variable_target_file is not None:
            with open(variable_target_file) as target_file:
                self.data_sql = json.load(target_file)
        # file read mode
        if source == 'presaved':
            if file_path is not None:
                _, file_extension = os.path.splitext(file_path)
                if file_extension not in self.io_function_switcher.keys():
                    raise IOError("Invalid file type {}!".format(file_extension))
                else:
                    index_col = dict()
                    io_reader = self.io_function_switcher.get(file_extension)
                    if file_extension in ['.csv', '.xlsx', '.xls']:
                        index_col['index_col'] = 0

                    self.data = io_reader(file_path, **index_col)
                    self.data_raw = self.data.copy()
                    if metadata is not None:
                        with open(metadata) as metadata_file:
                            self.metadata = json.load(metadata_file)
                        for variable in self.metadata['variables']:
                            variable_object = DataVariable(variable, self.metadata['variables'])
                            self.variable_name.append(variable_object)

            else:
                raise AttributeError('Pre-saved data target file name must be provided!')

        elif source == 'sklearn':
            from sklearn.datasets import load_boston, load_iris, load_diabetes, load_digits, load_linnerud, load_wine, \
                load_breast_cancer
            sklearn_data_switcher = {
                'boston': load_boston,
                'iris': load_iris,
                'diabetes': load_diabetes,
                'digits': load_digits,
                'linnerud': load_linnerud,
                'wine': load_wine,
                'breast_cancer': load_breast_cancer
            }  # functionality for dataset loading
            dataset_loader = sklearn_data_switcher.get(dataset_name)
            data_struct = dataset_loader()
            self.data = pd.DataFrame(data=data_struct.data, columns=data_struct.feature_names);
            self.data['target'] = data_struct.target
            self.data_raw = self.data.copy()
            self.variable_name_list = data_struct.feature_names

            self.metadata = dict()
            self.metadata['description'] = data_struct.DESCR
            self.metadata['variables'] = dict()
            for variable_name in self.variable_name_list:
                self.metadata['variables'][variable_name] = {'name': variable_name, 'source': 'sklearn'}
            if dataset_name in ['iris', 'breast_cancer', 'digits', 'wine']:
                self.metadata['output_classes'] = data_struct.target_names
            self.data_sql = self.metadata['variables']
            for variable in self.metadata['variables']:
                variable_object = DataVariable(variable, self.metadata['variables'])
                self.variable_name.append(variable_object)
        elif source == 'csv':
            self.data = pd.read_csv(file_path)
            self.data.set_index(self.data.columns[0], inplace=True)
            self.data_raw = self.data.copy()
            self.variable_name_list = self.data.columns[:-1]

            self.metadata = dict()
            self.metadata['variables'] = dict()
            for variable_name in self.variable_name_list:
                self.metadata['variables'][variable_name] = {'name': variable_name, 'source': 'csv'}
            self.data_sql = self.metadata['variables']
            for variable in self.metadata['variables']:
                variable_object = DataVariable(variable, self.metadata['variables'])
                self.variable_name.append(variable_object)

    def histogram(self, variable_list=None, bins=10):
        """Draw a histogram of the data and save to a pdf file"""
        if not self.data.empty:
            if variable_list is None or len(variable_list) < 1:
                warnings.warn("No variables specified! Nothing was created")
            elif any([var not in [variable.name for variable in self.variable_name] for var in variable_list]):
                warnings.warn("Invalid variable names found! Nothing was created")
            else:
                fig = plt.figure()
                ax = fig.subplots(nrows=len(variable_list), ncols=1)
                self.data.hist(column=variable_list, bins=bins, ax=ax)
                variables = '_'.join('{}'.format(var) for var in variable_list)
                fig.savefig('./figures/{}.pdf'.format(variables))

        else:
            warnings.warn("No data found! Nothing was created")

    def medication_timeseries(self):
        pass

    def normalize(self, variable_list=None, mode='mean_std', normalizing_values=None,
                  filter_column=None, apply_normalization=False, nan_flag=None, index_subset=None, nan_mode=None):
        """Normalize numerical variables by either subtracting mean and dividing by standard deviation (mode=mean_std)
        or subtracting minimum value and dividing by maximum value (mode=min_max). It returns normalizing values for
        saving (even if they are provided in which case they are not changed)"""
        if filter_column is not None:
            filt = np.arange(len(self.data.index))[self.data[filter_column]]
        else:
            filt = np.arange(len(self.data.index))

        if index_subset is not None:
            filt = index_subset

        if normalizing_values is not None:
            self.normalized_data = pd.DataFrame()
            for variable in variable_list:
                self.normalized_data[variable] = \
                    (self.data.iloc[filt][variable] - normalizing_values[variable][0]) \
                    / normalizing_values[variable][1]
        else:
            normalizing_values = dict()

            if mode == 'mean_std':
                for variable in variable_list:
                    mean_value = self.data.iloc[filt][variable].mean()
                    std_value = self.data.iloc[filt][variable].std()
                    if apply_normalization:
                        # self.normalized_data = self.data.copy()
                        self.normalized_data[variable] = (self.data.iloc[filt][variable] - mean_value) / std_value
                    normalizing_values[variable] = [mean_value, std_value]

            elif mode == 'min_max':
                for variable in variable_list:
                    min_value = self.data.iloc[filt][variable].min()
                    max_value = self.data.iloc[filt][variable].max()
                    if apply_normalization:
                        # self.normalized_data = self.data.copy()
                        self.normalized_data[variable] = (self.data.iloc[filt][variable] - min_value) / (max_value - min_value)
                    normalizing_values[variable] = [min_value, max_value - min_value]
            else:
                warnings.warn('Invalid normalization! Returned raw data')
        if nan_mode is not None and (nan_flag is not None and not nan_flag.empty and all(nan_flag.dtypes == bool)):
            self.normalized_data.iloc[nan_flag[variable_list].values] = 0.0
        return normalizing_values

    def resample_timeseries(self):
        pass

    def categorical_encoding(self, variable_list=None, nan_mode='include', nan_flag=None, index_subset=None):
        """Encodes categorical variables into one-hot vectors"""
        import itertools
        dummy_na = True if nan_mode == 'include' else False
        if index_subset is not None:
            filt = index_subset
        else:
            filt = np.arange(len(self.data.index))

        encoded_variables = list()
        encoded_nan_flag = pd.DataFrame()
        for variable in variable_list:
            one_hot_df = pd.get_dummies(self.data.iloc[filt][variable], dummy_na=dummy_na, prefix=variable)
            if nan_mode == 'flag' \
                    and variable + '_0.0' in one_hot_df.index \
                    and all(one_hot_df[variable + '_0.0'] == nan_flag[variable]):
                one_hot_df.drop(columns=[variable + '_0.0'], inplace=True)
            self.normalized_data = pd.concat([self.normalized_data,
                                              one_hot_df], axis=1)
            encoded_variables.append([column for column in one_hot_df.columns])
            if nan_mode == 'flag':  # repeat nan flag to fit the encoded variable
                encoded_nan_flag = pd.concat([encoded_nan_flag, nan_flag[[variable for column in one_hot_df.columns]]],
                                             axis=1)
        encoded_variables = list(itertools.chain.from_iterable(encoded_variables))
        if nan_flag is not None and not nan_flag.empty:  # repeat nan flag to fit the encoded variable
            not_encoded_variables = list(set(nan_flag.columns).difference(set(encoded_nan_flag.columns)))
            encoded_nan_flag.columns = encoded_variables
            encoded_nan_flag = pd.concat([nan_flag[not_encoded_variables], encoded_nan_flag], axis=1)
        return encoded_variables, encoded_nan_flag

    def filter_invalid(self, variable_list=None):
        """Filters data based on the outlier flags"""
        if variable_list is not None:
            for variable in variable_list:
                variable = '{}_outlier_flag'.format(variable)
                if variable not in self.data.columns:
                    continue
                filt = self.data[variable]
                self.data = self.data.loc[~filt]
                self.data.drop(columns=variable, inplace=True)

    def drop_useless_variables(self):
        """Removes columns with values that do not change and thus useless for ML training"""
        for variable in self.data.columns:
            if pd.unique(self.data[variable]).shape[0] < 2:
                self.data.drop(columns=variable, inplace=True)

    def extract_split(self, ratio, remove_from_original=False):
        """Extracts a part of the data set into a new data set. This is usually used to separate the test data"""
        if type(self.data.index) == pd.core.indexes.multi.MultiIndex:
            all_indices = pd.DataFrame(self.data.index.levels[0])
            test_indices = all_indices.sample(frac=ratio, random_state=1).values.flatten()
            self.test_data = self.data.loc[test_indices]
        else:
            self.test_data = self.data.sample(frac=ratio, random_state=1)
        if remove_from_original:
            self.data.drop(list(self.test_data.index), inplace=True)

    def split_flag(self, ratio, number_of_splits, stratify=None):
        """Create train-test splits and append flags that indicate each one"""
        if stratify is None:
            split_object = ShuffleSplit(n_splits=number_of_splits,
                                        train_size=ratio,
                                        test_size=1-ratio,
                                        random_state=0)
            if type(self.data.index) == pd.core.indexes.multi.MultiIndex:
                all_indices = self.data.index.get_level_values(0).drop_duplicates()
            else:
                all_indices = np.array(self.data.index)
            split_idx = 0
            for train_index, test_index in split_object.split(all_indices):
                self.data['train_split_{}'.format(split_idx)] = False
                self.data.loc[all_indices[train_index], 'train_split_{}'.format(split_idx)] = True
                split_idx += 1
        else:
            split_object = StratifiedShuffleSplit(n_splits=number_of_splits,
                                                  train_size=ratio,
                                                  test_size=1 - ratio,
                                                  random_state=0)
            if type(self.data.index) == pd.core.indexes.multi.MultiIndex:
                stratify.reset_index(inplace=True)
                stratify.drop_duplicates(subset=[stratify.columns[0]], inplace=True)
                stratify.drop(columns=stratify.columns[1], inplace=True)
                stratify.set_index(stratify.columns[0], inplace=True)
                all_indices = stratify.index.drop_duplicates()
            else:
                all_indices = np.array(self.data.index)
            split_idx = 0
            for train_index, test_index in split_object.split(all_indices, stratify.values):
                self.data['train_split_{}'.format(split_idx)] = False
                self.data.loc[all_indices[train_index], 'train_split_{}'.format(split_idx)] = True
                split_idx += 1

    def make_save_folder(self, folder_name):
        """Creates a folder for saving data"""
        if self.data_folder_name is None:
            os.mkdir('./data/{}/'.format(folder_name))
            self.data_folder_name = folder_name

    def save_data(self, metadata=None, type='csv', nan_flag=None):
        """Saves training/validation and test datasets and metadata"""
        if self.data_folder_name is None:
            time_now = datetime.now()
            current_time_str = time_now.strftime("%Y%m%d_%H%M")
            self.data_folder_name = 'saved_data_{}'.format(current_time_str)
            warnings.warn("No folder location was specified. Default name was assigned: {}".format(self.data_folder_name))
        training_file_name = 'training_validation_dataset'
        test_file_name = 'test_dataset'
        metadata_file_name = 'metadata'
        variable_names = [variable.name for variable in self.variable_name]
        if metadata is None:
            metadata = dict()
        metadata['variables'] = dict()
        for variable in variable_names:
            metadata['variables'][variable] = self.data_sql[variable]
        if type == 'csv':
            self.data.to_csv('./data/{}/{}.csv'.format(self.data_folder_name, training_file_name))
        elif type == 'pickle':
            self.data.to_pickle('./data/{}/{}.pkl'.format(self.data_folder_name, training_file_name))
        if self.test_data is not None:
            if type == 'csv':
                self.test_data.to_csv('./data/{}/{}.csv'.format(self.data_folder_name, test_file_name))
            elif type == 'pickle':
                self.test_data.to_pickle('./data/{}/{}.pkl'.format(self.data_folder_name, test_file_name))
        with open('./data/{}/{}.json'.format(self.data_folder_name, metadata_file_name), 'w') as outfile:
            json.dump(metadata, outfile)
        if nan_flag is not None:
            nan_flag.to_csv('./data/{}/{}.csv'.format(self.data_folder_name, 'nan_flag'))

    def remove_nulls(self, variable_list):
        """Removes rows with null values from certain columns"""
        for variable in variable_list:
            filt = self.data[variable].isnull()
            self.data = self.data.loc[~filt]

    def default_impute(self, variable_list=None, missing_flag=None):
        """This applies the default imputation on each variable.
        The default imputation is saved in the data target file"""
        for idx, variable in enumerate(variable_list):
            variable_properties = next((x for x in self.variable_name if x.name == variable))
            if 'impute_mode' in variable_properties.variable_target.keys():
                if not self.data.empty:
                    if missing_flag is not None:
                        missing_flag_var = missing_flag[variable]
                        data_series = self.data.loc[~missing_flag_var, variable]
                    else:
                        data_series = self.data[variable]
                    impute_function = {'mean': data_series.mean,
                                       'median': data_series.median}
                    impute = impute_function.get(variable_properties.variable_target['impute_mode'])
                    if missing_flag is not None:
                        impute_value = impute()
                        self.data.loc[missing_flag_var, variable] = impute_value
                    else:
                        self.data[variable].fillna(impute(), inplace=True)

    def mean_impute(self, variable_list=None, missing_flag=None):
        """This applies the default imputation on each variable.
        The default imputation is saved in the data target file"""
        for idx, variable in enumerate(variable_list):
            if not self.data.empty:
                data_series = self.data[variable]
                impute = data_series.mean
                self.data[variable].fillna(impute(), inplace=True)

    def flag_missing(self, variable_list=None):
        """This flags the missing values and replaces all of those values by zeros"""
        nan_flag = self.data[variable_list].isna()
        self.data.fillna(0, inplace=True)
        return nan_flag

    def flag_missing_keep_nan(self, variable_list=None):
        """This flags the missing values and replaces all of those values by zeros"""
        nan_flag = self.data[variable_list].isna()
        return nan_flag

    def extract_subset(self, ratio, stratify=False):
        """Extracts a subset of data"""
        all_indices = np.array(self.data.index)
        if stratify is None:
            split_object = ShuffleSplit(n_splits=1,
                                        train_size=ratio,
                                        test_size=1 - ratio,
                                        random_state=0)

            train_index, _ = next(split_object.split(all_indices))

        else:
            split_object = StratifiedShuffleSplit(n_splits=1,
                                                  train_size=ratio,
                                                  test_size=1 - ratio,
                                                  random_state=0)
            train_index, _ = next(split_object.split(all_indices, stratify.values))
        return train_index

    def remove_quantile(self, quantile=1.0, mode='bigger', variable_list=None):
        if mode == 'bigger':
            quantile_flag = self.data[variable_list] > self.data[variable_list].quantile(quantile)
        elif mode == 'smaller':
            quantile_flag = self.data[variable_list] < self.data[variable_list].quantile(quantile)
        self.data[quantile_flag] = np.nan

    def add_noise(self, std_range=[0.0, 1.0], variable_list=None):
        noise_std = pd.DataFrame(data=np.random.uniform(std_range[0], std_range[1],
                                                        size=self.data[variable_list].shape),
                                 index=self.data.index,
                                 columns=variable_list)
        noise_std = noise_std * self.data[variable_list].std()
        noise_values = np.random.normal(0.0, noise_std.values)
        self.data[variable_list] = self.data[variable_list] + noise_values
        return noise_std / noise_std.values.max()
