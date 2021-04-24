# Atividade final do módulo - EX04
# Autor: Newmar Wegner, Paulo, Kleberson
# Date: 23/05/2021

import pandas as pd
from sklearn import preprocessing

# Function to normalize data considering Min Max Scaler sklearn
def normalize_data_minmax(df):
    normalizer = preprocessing.MinMaxScaler()
    
    return normalizer.fit(df), pd.DataFrame(normalizer.fit_transform(df))


# Function to normalize dummy data
def normalize_data_dummies(df):
    return pd.get_dummies(df, prefix=df.columns)


if __name__ == '__main__':
    df = pd.read_csv('../datasets/dados_normalizar.csv', sep=';')
    df_num = df.drop(columns=['sexo'])
    df_categories = pd.DataFrame(df['sexo'])
    model, df_num_norm_minmax = normalize_data_minmax(df_num)
    df_categories_norm = normalize_data_dummies(df_categories)
    normalize_dataframe = pd.concat([df_num_norm_minmax, df_categories_norm], axis=1)
    print(normalize_dataframe)
