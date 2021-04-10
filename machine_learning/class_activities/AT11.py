# Atividade em aula - AT11
# Autor: Newmar Wegner
# Date: 10/04/2021

"""
ATIVIDADE
•Utilizando arquivo de dados Bank
    1. Indutor e modelo
        –Será necessário normalizar os dados. Portanto, o modelo normalizador deve ser salvo para uso com as novas
        instâncias
        –Verificar necessidade de balanceamento
        –Obter um modelo de ML com RandomForest
        –Obter um modelo de ML com outro indutor, à sua escolha, exceto Redes Neurais
        –Atenção: Se for utilizar SVM, foco no kernel linear
        –Registrar a acurácia dos modelo obtidos (lembra de dividir a base em train/test)
    2. Módulo de inferência
        –Criar um novo programa em Py
        –Simular uma nova instância
        –Classificar com os dois modelos obtidos
"""

from pickle import dump
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Function to normalize data considering Min Max Scaler sklearn
def normalize_data_minmax(df):
    normalizer = preprocessing.MinMaxScaler()
    
    return normalizer.fit(df), normalizer.fit_transform(df)


# Function to normalize dummy data
def normalize_data_dummies(df):
    return pd.get_dummies(df, prefix=df.columns)


# function to create categorical and numerical dataset
def create_catnum(df, dummy_columns, class_column):
    categorical = df[dummy_columns]
    numerical = df.drop(columns=dummy_columns + class_column, axis=1)
    
    return categorical, numerical


# Function to Balance dataset
def balance_data(final_data,class_column):
    x, y = final_data.iloc[:, :-1], final_data[class_column]
    oversample = SMOTE()
    
    return oversample.fit_resample(x, y)

# Function to save model
def save_model(model):
    return dump(model, open(f'../datasets/model.pkl', 'wb'))

if __name__ == '__main__':
    # Input parameters
    df = pd.read_csv(f'../datasets/bank.csv', sep=';')
    dummy_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    class_column = ['y', ]
    
    # Creating categorical and numerical dataset
    categorical, numerical = create_catnum(df, dummy_columns, class_column)
    
    # Get dummies to categorical dataset
    norm_categorical = normalize_data_dummies(categorical)
    
    # Join norm_categorical with numerical dataset to normalize with minmax scaler
    dados = numerical.join(norm_categorical, how='left')
    model, df_transformed = normalize_data_minmax(dados)
    
    # Saving normalizer
    dump(model, open(f'../datasets/MinMaxScaler_bank.pkl', 'wb'))
    
    # Saving final normalizer dataset
    final_data = dados.join(df['y'], how='left')
    final_data.to_csv(f"../datasets/Bank_normalize.csv", index=False, sep=';')

    # Get balanced dataset
    x_oversample, y_oversample = balance_data(final_data, class_column)
 
    # Slice test and train data
    X_train, X_test, Y_train, Y_test = train_test_split(x_oversample, y_oversample, test_size=0.3)
   
    # Get model
    rf_bank = RandomForestClassifier().fit(X_train, Y_train.values.ravel())

    # Test predict
    y_predict = rf_bank.predict(X_test)
    print(y_predict)

    # Accuracy: compare y_predict with Y_test
    print(f'Global Accuracy RandomForest: {round(metrics.accuracy_score(Y_test, y_predict), 3)}')

    # Save model
    save_model(rf_bank)

    # TODO: Realizar outro modelo da minha escolha
