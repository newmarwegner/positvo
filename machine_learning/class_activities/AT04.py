# Atividade em aula - AT04
# Autor: Newmar Wegner
# Date: 20/03/2021

import os
import pandas as pd

# Function to normalize dummy data
def normalize_data_dummies(df):
    
    return pd.get_dummies(df, prefix=df.columns)

if __name__ == '__main__':
    df = pd.read_csv(os.path.dirname(__file__) + '/../datasets/vote.csv', sep=',')
    df_result = pd.DataFrame(normalize_data_dummies(df))
    df_result.to_csv('results/vote_norm.csv', index=False, sep=',')
