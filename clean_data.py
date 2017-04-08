#For detecting missing values and outliers
import pandas as pd

def show_missing (col):
    '''
    This function gets total missing values for a column
    :param df: column from pandas dataframe
    :return: number of missing values for the column
    '''
    return sum(col.isnull())

def return_outliers (df,col):
    '''
    This function uses IQR rule to detect and return
    :param df: data frame
    :param col: field from the data frame with quotes
    :return: records that are outlier (i.e, those falling outside the IQR range)
    '''

    q1 = df[col].quantile(0.25)
    med = df[col].median()
    q3 = df[col].quantile(0.75)
    iq_range = q3 - q1
    last_q = df[(df[col] > (q3 + (1.5 * iq_range)))]
    first_q = df[(df[col] < (q1 - (1.5 * iq_range)))]

    for i in first_q.iterrows():
        print "-------------------------------------------------------"
        print i

    for i in last_q.iterrows():
        print "-------------------------------------------------------"
        print i

def show_error(col):
    '''
    This function gets the median length of a value and return values that are not in IQR range
    :param col: column from a data frame
    :return: print values not in IQR range
    '''

    import numpy as np
    length = []
    for i in col:
        length.append(len(i))

    q1 = np.percentile(length,25)
    med_len = np.median(length)
    q3 = np.percentile(length,75)
    iq_range = q3 - q1

    for i in col:
        if len(i) > (q3 + (1.5 * iq_range)) or len(i) < (q1 - (1.5 * iq_range)):
            print i


