# Atividade em aula - AT03
# Autor: Newmar Wegner
# Date: 20/03/2021

import os
from pickle import dump
import pandas as pd
from sklearn import preprocessing

# Function to normalize data considering Min Max Scaler sklearn
def normalize_data_minmax(df):
    normalizer = preprocessing.MinMaxScaler()
    
    return normalizer.fit(df), normalizer.fit_transform(df)

if __name__ == '__main__':
    df = pd.read_csv(os.path.dirname(__file__) + '/../datasets/BostonHousing.csv', sep=';')
    model, df_transformed = normalize_data_minmax(df)
    dump(model, open('results/MinMaxScaler_model_boston.pkl','wb'))
    df_result = pd.DataFrame(df_transformed, columns=df.columns)
    df_result.to_csv('results/BostonHousing_norm.csv', index=False, sep=';')
