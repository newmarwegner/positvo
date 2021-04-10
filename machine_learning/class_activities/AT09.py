# Atividade em aula - AT09
# Autor: Newmar Wegner
# Date: 10/04/2021

"""
ATIVIDADE
1. Balancear dado fertility
2. Obter uma Random Forest
"""

import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter


# Function to open database
def open_csv(data):
    return pd.read_csv(f'../datasets/{data}')


# Function to slice attributes and classes
def get_x_y(data, class_field):
    x = data.drop(columns=[class_field])
    y = data[class_field]
    
    return x, y


# Function to Balance database
def balance_data(x, y):
    oversample = SMOTE()
    
    return oversample.fit_resample(x, y)


if __name__ == '__main__':
    # define the name of data
    data = open_csv('fertility.txt')
    # define class field
    class_field = 'Output'
    # get attributes and class tables
    x, y = get_x_y(data, class_field)
    # Counter class fields
    print(Counter(y))
    # Balanced data
    x_oversample, y_oversample = balance_data(x, y)
    print(Counter(y_oversample))
