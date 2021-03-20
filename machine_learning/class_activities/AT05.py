# Atividade em aula - AT05
# Autor: Newmar Wegner
# Date: 20/03/2021

'''
ATIVIDADE
1. Em arquivo / programa : abrir dados, obter o modelo normalizador e salvar em disco /// executado na AT03
2. Em outro arquivo / programa:
       abrir o modelo normalizador,
       simular novas instâncias,
       imprimir os dados das novas instâncias (originais e normalizados)
       normalizar as novas instâncias
'''

import os
import numpy as np
import pandas as pd
from pickle import load
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def norm_new_dataset(model_trasform, new_values):
    return model_transform.transform(pd.DataFrame(new_values))


if __name__ == '__main__':
    df = pd.read_csv(os.path.dirname(__file__) + '/../datasets/BostonHousing.csv', sep=';')
    print(df.describe())
    model_transform = load(open('results/MinMaxScaler_model_boston.pkl', 'rb'))
    new_values = np.array([[0.03237, 0, 2.18, 0, 0.458, 6.998, 45.8, 6.0622, 3, 222, 18.7, 394.63, 2.94, 33.4],
                           ])
    print(pd.DataFrame(new_values, columns=df.columns))
    print(pd.DataFrame(norm_new_dataset(model_transform, new_values), columns=df.columns))
