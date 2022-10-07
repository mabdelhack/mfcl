import pandas as pd
import numpy as np
from EMRData.utils import apply_operation_list


class DataVariable(object):
    """This creates a pandas column dataframe for each variable and stores its properties"""

    def __init__(self, variable, data_sql):

        self.data_sql = data_sql
        self.variable_target = self.data_sql.get(variable)
        self.data = pd.DataFrame(columns=[self.variable_target['name']])
        self.name = variable
        self.ID = self.data_sql.get('ID')

    def get_data_sql(self, sql_data_pointer, id_list=None, pipeline=True):
        """Get variable from database and preprocess it if needed"""
        print('Loading {} variable'.format(self.variable_target['name']))
        sql_data = sql_data_pointer.sql_query(self.variable_target, self.ID, id_list=id_list)
        if self.variable_target['category'] == 'interop':
            # For interop, I add time as an additional index
            sql_data.dropna(inplace=True)
            sql_data['Time'] = (pd.to_timedelta(sql_data['Time']).dt.total_seconds() / 60).astype('int16')
            sql_data = sql_data.loc[sql_data['Time'] >= 0]
            sql_data.set_index('Time', append=True, inplace=True)
        if pipeline:
            sql_data = self.apply_operation(sql_data_pointer.source, sql_data)
            if self.variable_target['category'] == 'medication':
                # For medication data, there are four columns indicating amount, start and end times, and whether
                # it's a one dose medication or periodic
                second_level_heading = ['Amount', 'Start', 'End', 'Measurement_Unit']
                first_level_heading = [self.name] * len(second_level_heading)
                # nested column format of [medication_name]:[column_name]
                nested_column_names = ['{}:{}'.format(medication, info)
                                       for (medication, info) in zip(first_level_heading, second_level_heading)]
                columns = {key: value for (key, value) in zip(sql_data.columns, nested_column_names)}
                sql_data.rename(columns=columns, inplace=True)
            else:
                # Rename column to standard
                sql_data.rename(columns={sql_data.columns[0]: self.name}, inplace=True)
            sql_data = self.flag_outliers(sql_data)
            sql_data = self.out_of_range_to_nan(sql_data)
        return sql_data

    def apply_operation(self, source, sql_data):
        """Applies preprocessing to data"""
        operation_location = '{}_operation'.format(source)
        if operation_location in self.variable_target:
            sql_data = apply_operation_list(sql_data, self.variable_target[operation_location])
        return sql_data

    def flag_outliers(self, sql_data):
        """Creates a column that flags outliers for deletion"""
        if 'acceptable_range' in self.variable_target:
            nan_flag_name = '{}_outlier_flag'.format(self.name)
            sql_data[nan_flag_name] = (sql_data < self.variable_target['acceptable_range'][0]) \
                                      | (sql_data > self.variable_target['acceptable_range'][1])
        return sql_data

    def out_of_range_to_nan(self, sql_data):
        """Converts any out-of-range values to Nan and applies imputation"""
        if 'non_null_range' in self.variable_target:
            sql_data.loc[sql_data[self.name] < self.variable_target['non_null_range'][0], self.name] = None
            sql_data.loc[sql_data[self.name] > self.variable_target['non_null_range'][1], self.name] = None

        return sql_data
