import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import ShuffleSplit, StratifiedShuffleSplit


class SimulatedDataGenerator(object):
    """This class creates simulation data"""
    def __init__(self, sample_shape, missing_portion=0, noise_variance=1, fill_na=np.nan,
                 positive_correlated=0.25, negative_correlated=0.25, uncorrelated=0.5,
                 flip_probability=0.0):
        assert positive_correlated + negative_correlated + uncorrelated == 1.0
        self.sample_shape = sample_shape
        self.missing_portion = missing_portion
        self.noise_variance = noise_variance
        self.fill_na = fill_na
        self.data_proportions = [positive_correlated, negative_correlated, uncorrelated]

    def generate_data_logistic(self, number_of_samples, proportion_positive=0.5, min_mult=0.2, max_mult=1.0,
                               flip_probability=0.0):
        """Generates data that would be used for a classification task where the target is either 1 or 0"""
        label = np.random.rand(number_of_samples)
        flip_vector = np.random.rand(number_of_samples)
        flip_vector = flip_vector < flip_probability
        flip_vector = flip_vector.astype(int)
        y = label < proportion_positive
        y = y.astype(int)
        x = np.random.normal(loc=0.0, scale=self.noise_variance, size=(number_of_samples, self.sample_shape))
        data_edges = np.cumsum(self.data_proportions)

        # positively correlated
        x[:, :int(data_edges[0] * self.sample_shape)] = x[:, :int(data_edges[0]*self.sample_shape)] + \
            np.matmul(np.expand_dims(y, axis=1),
                      np.abs(np.random.uniform(low=min_mult, high=max_mult,
                                               size=(1, int(self.sample_shape*self.data_proportions[0]))
                                               )
                             )
                      )
        # negatively correlated
        x[:, int(data_edges[0] * self.sample_shape):int(data_edges[1] * self.sample_shape)] = \
            x[:, int(data_edges[0] * self.sample_shape):int(data_edges[1] * self.sample_shape)] - \
            np.matmul(np.expand_dims(y, axis=1),
                      np.abs(np.random.uniform(low=min_mult, high=max_mult,
                                               size=(1, int(self.sample_shape*self.data_proportions[1]))
                                               )
                             )
                      )
        missing_flag = np.zeros((number_of_samples, self.sample_shape))
        if self.missing_portion > 0:
            missing_probability = np.random.uniform(low=0.0, high=1.0, size=missing_flag.shape)
            missing_flag = missing_probability < self.missing_portion
            missing_flag = missing_flag.astype(int)
        x = np.ma.array(x, mask=missing_flag)
        x = x.filled(self.fill_na)
        # shuffling samples
        sample_order = np.arange(number_of_samples)
        np.random.shuffle(sample_order)
        x = x[sample_order]
        missing_flag = missing_flag[sample_order]
        y = y[sample_order]
        y = np.abs(y - flip_vector)
        return x, missing_flag, y

    @staticmethod
    def generate_missing(x, missing_portion, fill_na):
        """This generates a version of data already generated with missing values"""
        missing_flag = np.zeros(x.shape)
        if missing_portion > 0:
            missing_probability = np.random.uniform(low=0.0, high=1.0, size=x.shape)
            missing_flag = missing_probability < missing_portion
            all_missing = x.shape[0] - sum(missing_flag) == 0
            if any(all_missing):
                missing_flag[0, all_missing] = False
            missing_flag = missing_flag.astype(int)
        x = np.ma.array(x, mask=missing_flag)
        x = x.filled(fill_na)
        return x, missing_flag

    @staticmethod
    def save_data(data, folder_name, split=None):
        from os import mkdir, path
        x = data[0]
        y = data[1]
        if len(data) > 2:
            missing_flag = data[2]
        else:
            missing_flag = np.empty(shape=(x.shape[0], 0))
        index_column = np.arange(x.shape[0])
        column_names = ['var_{}'.format(n) for n in range(x.shape[1])]
        if len(data) > 2:
            for n in range(missing_flag.shape[1]):
                column_names.append('missing_flag_{}'.format(n))
        column_names.insert(0, 'ID')
        data_concatenated = np.concatenate((np.expand_dims(index_column, axis=1),
                                            x, missing_flag,
                                            np.expand_dims(y, axis=1)), axis=1)
        column_names.append('output')
        df = pd.DataFrame(data=data_concatenated, columns=column_names)
        df.set_index(['ID'], inplace=True)
        if split is not None:

            df = split_flag(df, split['ratio'], number_of_splits=split['number_of_splits'], stratify=df['output'])

        mkdir(folder_name)
        to_save_file = path.join(folder_name, 'training_validation_dataset.csv')
        df.to_csv(to_save_file)
        return df

    @staticmethod
    def calculate_normalization_values(data, variable_list=None, mode='mean_std', filter_column=None):
        """Normalize numerical variables by either subtracting mean and dividing by standard deviation (mode=mean_std)
        or subtracting minimum value and dividing by maximum value (mode=min_max). It returns normalizing values for
        saving (even if they are provided in which case they are not changed)"""
        if filter_column is not None:
            filt = data[filter_column]
        else:
            filt = data.index

        normalizing_values = dict()

        if mode == 'mean_std':
            for variable in variable_list:
                mean_value = data.loc[filt, variable].mean()
                std_value = data.loc[filt, variable].std()

                normalizing_values[variable] = [mean_value, std_value]

        elif mode == 'min_max':
            for variable in variable_list:
                min_value = data.loc[filt, variable].min()
                max_value = data.loc[filt, variable].max()

                normalizing_values[variable] = [min_value, max_value - min_value]
        else:
            warnings.warn('Invalid normalization! Returned raw data')
        return normalizing_values


def split_flag(data, ratio, number_of_splits, stratify=None):
    """Create train-test splits and append flags that indicate each one"""
    if stratify is None:
        split_object = ShuffleSplit(n_splits=number_of_splits,
                                    train_size=ratio,
                                    test_size=1 - ratio,
                                    random_state=0)
        all_indices = np.array(data.index)
        split_idx = 0
        for train_index, test_index in split_object.split(all_indices):
            data['train_split_{}'.format(split_idx)] = False
            data.loc[data.index[train_index], 'train_split_{}'.format(split_idx)] = True
            split_idx += 1
    else:
        split_object = StratifiedShuffleSplit(n_splits=number_of_splits,
                                              train_size=ratio,
                                              test_size=1 - ratio,
                                              random_state=0)
        all_indices = np.array(data.index)
        split_idx = 0
        for train_index, test_index in split_object.split(all_indices, stratify.values):
            data['train_split_{}'.format(split_idx)] = False
            data.loc[data.index[train_index], 'train_split_{}'.format(split_idx)] = True
            split_idx += 1
    return data


