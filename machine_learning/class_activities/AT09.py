# Atividade em aula - AT09
# Autor: Newmar Wegner
# Date: 10/04/2021

"""
ATIVIDADE
1. Balancear dado fertility
2. Normalizar (como a fertility já está normalizada não é necessario)
3. Segmentar dados de treino e testes
4. Obter o modelo Random Forest
5. Gerar a accuracia global
6. Salvar o modelo
"""

from pickle import dump
import pandas as pd
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


# Function to open dataset
def open_csv(data):
    return pd.read_csv(f'../datasets/{data}')


# Function to slice attributes and classes
def get_x_y(data, class_field):
    x = data.drop(columns=[class_field])
    y = data[class_field]
    
    return x, y


# Function to Balance dataset
def balance_data(x, y):
    oversample = SMOTE()
    
    return oversample.fit_resample(x, y)


# Function to save model
def save_model(model):
    return dump(model, open(f'../datasets/model.pkl', 'wb'))


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
    # Normalize dataset if necessary
    
    # Slice test and train data
    X_train, X_test, Y_train, Y_test = train_test_split(x_oversample, y_oversample, test_size=0.3)
    
    # Get model
    rf_fertility = RandomForestClassifier().fit(X_train, Y_train)
    
    # Test predict
    y_predict = rf_fertility.predict(X_test)
    print(y_predict)
    
    # Accuracy: compare y_predict with Y_test
    print(f'Global Accuracy: {round(metrics.accuracy_score(Y_test, y_predict), 3)}')
    
    # Save model
    save_model(rf_fertility)
