import numpy as np
import pandas as pd


def apply_operation_list(dataframe, operation_list):
    """Apply the list of operations to the dataframe"""
    result = pd.DataFrame(index=dataframe.index)

    for operation_pair in operation_list:
        operation_name = operation_pair[0]
        if len(operation_pair) > 1:
            additional_arguments = operation_pair[1]
            operation = "{}(dataframe, additional_arguments)".format(operation_name)
        else:
            operation = "{}(dataframe)".format(operation_name)
        dataframe = eval(operation)
    return dataframe


def datetime_difference(dataframe):
    days_in_a_year = 365.2425
    return dataframe[dataframe.columns[0]] - dataframe[dataframe.columns[1]] / days_in_a_year


def string_to_value(dataframe, unit_string):
    medication_string = dataframe[dataframe.columns[0]]
    medication_string_split_mg = medication_string.split(unit_string)
    medication_string_split_space = medication_string_split_mg[0].split(' ')
    resulting_value = float(medication_string_split_space[-1])
    return resulting_value


def ideal_body_weight(dataframe):
    sex = dataframe[dataframe.columns[1]]
    height = string_to_value(dataframe[dataframe.columns[0]], 'cm')
    ideal_body_weight = 2.3 / 2.54 * (height - 152.4)
    if sex == 'Male':
       ideal_body_weight = ideal_body_weight + 50
    else:
       ideal_body_weight = ideal_body_weight + 45.5
    return ideal_body_weight


def bmi(dataframe):
    weight = string_to_value(dataframe[dataframe.columns[1]], 'kg')
    height = string_to_value(dataframe[dataframe.columns[0]], 'cm')
    bmi = weight / ( (height / 100.0) ^ 2 )
    return bmi

def keep_higher(dataframe):
    result = max(dataframe[dataframe.columns[0]], dataframe[dataframe.columns[1]])
    return result


def split_systolic_pressure(dataframe):
    medication_string = dataframe[dataframe.columns[0]]
    medication_string_split_slash = medication_string.split('/')
    resulting_value = float(medication_string_split_slash[0])
    return resulting_value


def split_diastolic_pressure(dataframe):
    medication_string = dataframe[dataframe.columns[0]]
    medication_string_split_slash = medication_string.split('/')
    resulting_value = float(medication_string_split_slash[-1])
    return resulting_value


def map_values(dataframe, mapping):
    for map_pair in mapping:
        map_pair = [np.nan if x is None else x for x in map_pair]
        dataframe.replace(map_pair[0], map_pair[1], inplace=True)
    return dataframe


def text_to_binary(dataframe):
    pass


def categorical_to_onehot(dataframe):
    pass


def categorical_to_binary(dataframe, mapping):
    from math import isnan
    result = pd.DataFrame(columns=[dataframe.columns[0]])
    result[dataframe.columns[0]] = \
        dataframe[dataframe.columns[0]].apply(lambda x: 1. if x in mapping else (0. if not isnan(x) else None))
    return result


def periodic_flag(dataframe):
    periodic_check = lambda x: 1. if (x is not None and '/' in x) else 0.
    dataframe[dataframe.columns[-1]] = dataframe[dataframe.columns[-1]].apply(periodic_check)
    return dataframe


def time_series_till(dataframe):
    dataframe.dropna(inplace=True)
    dataframe[dataframe.columns[0]] = (pd.to_timedelta(dataframe[dataframe.columns[0]]
                                                       ).dt.total_seconds() / 60).astype('int16')
    dataframe[dataframe.columns[0]] = dataframe[dataframe.columns[0]].map(lambda x: list(range(x)))
    dataframe = pd.DataFrame([(index, value) for (index, values) in dataframe[dataframe.columns[0]].iteritems() \
                              for value in values], columns=['PatientID', dataframe.columns[0]])
    dataframe.set_index('PatientID', inplace=True)
    dataframe['Time'] = dataframe[dataframe.columns[0]]
    dataframe.set_index('Time', append=True, inplace=True)
    return dataframe


def medication_multiindex(dataframe):  # deprecated
    return dataframe


def medication_rescale(dataframe):
    dataframe.apply(lambda row : medication_convert_units(standard_unit,
                                                          dataframe['Amount'],
                                                          dataframe['Unit'],
                                                          dataframe['Weight'],
                                                          ))


def medication_convert_units(standard_unit, amount, unit, patient_weight, time_portion):

    volume_conversion = pd.DataFrame(np.array([[1, 1, 1000, 1, 1e12],
                                               [1, 1, 1000, 1, 1e12],
                                               [1e-3, 1e-3, 1, 1e-3, 1e9],
                                               [1, 1, 1000, 1, 1e12],
                                               [1e-12, 1e-12, 1e-9, 1e-12, 1]]),
                                     columns=['ml', 'cc', 'mcl', '10 ml', 'fl'],
                                     index=['ml', 'cc', 'mcl', '10 ml', 'fl'])

    weight_conversion = pd.DataFrame(np.array([[1, 1e-3, 1000, 1e6, 2.2e-6],
                                               [1000, 1, 1e6, 1e9, 2.2e-3],
                                               [1e-3, 1e-6, 1, 1e3, 2.2e-9],
                                               [1e-6, 1e-9, 1e-3, 1, 2.2e-12],
                                               [0.45e6, 0.45e3, 0.45e9, 0.45e12, 1]]),
                                     columns=['mg', 'g', 'mcg', 'ng', 'lbs'],
                                     index=['mg', 'g', 'mcg', 'ng', 'lbs'])

    u_conversion = pd.DataFrame(np.array([[1, 1, 1e3],
                                          [1, 1, 1e3],
                                          [1e-3, 1e-3, 1]]),
                                columns=['u', 'iu', 'miu'],
                                index=['u', 'iu', 'miu'])

    time_conversion = {'h': 60.0, 'min': 1.0, 'day': 1440.0, 'week': 10080.0}

    # start with no conversion
    multiplier = 1.0

    standard_unit = standard_unit.lower()

    unit = unit.lower()
    unit_elements = unit.split('/')
    # check if patient weight exists
    if 'kg' in unit_elements:
        multiplier = multiplier * patient_weight

    # check the amount unit
    if unit_elements[0] in ['ml', 'cc', 'mcl', 'fl', '10 ml']:
        multiplier = multiplier * volume_conversion[standard_unit][unit_elements[0]]
    elif unit_elements[0] in ['mg', 'g', 'mcg', 'ng', 'lbs']:
        if standard_unit == 'puffs':
            standard_unit_replacement = 'mg'
        else:
            standard_unit_replacement = standard_unit
        multiplier = multiplier * weight_conversion[standard_unit_replacement][unit_elements[0]]
    elif unit_elements[0] in ['u', 'iu', 'miu']:
        multiplier = multiplier * u_conversion[standard_unit][unit_elements[0]]
    elif unit_elements[0] in ['pack', 'drop']:  # Ignore values with weird units
        multiplier = 0

    if unit_elements[-1] in ['h', 'min', 'day', 'week']:
        multiplier = multiplier / time_conversion[unit_elements[-1]]
        multiplier = multiplier * time_portion

    converted_amount = amount * multiplier

    return converted_amount


def to_categorical(dataframe, categories=None):
    if categories is not None:
        category_dtype = pd.api.types.CategoricalDtype(categories=categories)
        dataframe = dataframe.astype(category_dtype)
    else:
        dataframe = dataframe.astype('category')
    dataframe[dataframe.columns[0]] = dataframe[dataframe.columns[0]].cat.codes
    return dataframe
