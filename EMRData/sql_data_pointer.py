import pandas as pd
import warnings
import json


class SqlDataPointer(object):
    """This object is for querying different databases for medical records"""
    def __init__(self, source='Epic'):
        self.source = source
        self.target_set = None
        self.cnxn = None

    def connect(self):
        """Connecting to the database"""
        if self.source == 'Epic':
            import pyodbc
            with open('./epic_login.json') as login_file:
                login_credentials = json.load(login_file)
            driver = login_credentials['driver']
            server = login_credentials['server']
            database = login_credentials['database']
            self.cnxn = pyodbc.connect(r'Driver={{{}}};Server={};Database={};Trusted_Connection=yes;'.format(driver, server, database))
            self.variable_set = 'epic_targets'
        elif self.source == 'heartdata':
            import mysql.connector
            with open('./heartdata_login.json') as login_file:
                login_credentials = json.load(login_file)
            user = login_credentials['user']
            password = login_credentials['password']
            host = login_credentials['host']
            database = login_credentials['database']
            self.cnxn = mysql.connector.connect(user=user, password=password, host=host,
                                                database=database)
            self.target_set = 'heartdata_targets'

    def disconnect(self):
        """Disconnect the database"""
        if self.cnxn is not None and self.cnxn.is_connected():
            self.cnxn.close()
            self.cnxn = None

    def sql_query(self, sql_variable_target=None, sql_id_target=None, id_list=None):
        """"Actual querying function, it parses the data targets to build SQL statement and query the database"""
        if sql_variable_target is None:
            warnings.warn('Invalid variable name! Returning None.')
            result_df = None
        else:
            sql_variable_call = sql_variable_target[self.target_set]
            if sql_variable_call is None:
                warnings.warn('No pointer in database available. Returning None.')
                result_df = None
            else:
                sql_id_call = sql_id_target[self.target_set]

                query = select_query(sql_variable_call, sql_id_call)

                if id_list is not None:
                    query = "{} {}".format(query, code_statement(sql_id_call['table'],
                                                                 sql_id_call['column'], id_list, 'WHERE'))

                query = query + ';'
                result_df = pd.read_sql_query(query, self.cnxn, index_col=sql_id_call['column'])

        return result_df


def select_query(sql_variable_call, sql_id_call):
    """Create a SELECT SQL statement"""
    # Select part (always query using Patient ID as the main variable
    query = "SELECT {}.{},{} FROM".format(sql_id_call['table'], sql_id_call['column'],
                                          ','.join('{0}.{1}'.format(variable['table'], variable['column'])
                                                   for variable in sql_variable_call))
    joiner, _ = join_statement(sql_variable_call, sql_id_call)
    query = '{} {}'.format(query, joiner)

    return query


def join_statement(sql_variable_call, sql_id_call):
    """Create a JOIN SQL fragment"""
    if len(sql_variable_call) > 0:
        join_query, tables_set = join_statement(sql_variable_call[1:], sql_id_call)
    else:
        tables_set = [sql_id_call["table"]]
        join_query = sql_id_call["table"]
        return join_query, tables_set
    code_word = 'WHERE'
    join_flag = sql_variable_call[-1]["table"] not in tables_set
    if join_flag:
        tables_set.append(sql_variable_call[-1]["table"])
        join_query = "({} LEFT JOIN {} ON {}.{} = {}.{})".format(join_query, sql_variable_call[0]["table"],
                                                                 sql_id_call["table"], sql_id_call["column"],
                                                                 sql_variable_call[0]["table"], sql_id_call["column"])
        code_word = 'AND'
    if (len(tables_set) == 1 or join_flag)\
            and 'code' in sql_variable_call[-1]:
        join_query = "{} {})".format(join_query[:-1], code_statement(sql_variable_call[0]['table'],
                                                                     sql_variable_call[0]['code_column'],
                                                                     sql_variable_call[0]['code'], code_word))

    return join_query, tables_set


def code_statement(code_table, code_column, codes, sql_conditional):
    """Appends different codes for filtering data into the SQL statement using conditionals"""
    condition_set = '({})'.format(','.join('"{0}"'.format(cond) for cond in codes))
    sql_statement = "{} {}.{} IN {}".format(sql_conditional, code_table, code_column, condition_set)
    return sql_statement
