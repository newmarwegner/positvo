# Atividade final do módulo - Lista 02 - EX01
# Autor: Newmar Wegner, Paulo, Kleberson
# Date: 23/05/2021

# Function to normalize data considering Min Max Scaler sklearn
import pandas as pd
from pickle import load
from sklearn import preprocessing

# Function to normalize input
def normalize_data_minmax(df):
    normalizer = preprocessing.MinMaxScaler()
    
    return normalizer.fit(df), normalizer.fit_transform(df)

if __name__ == '__main__':
    input_data = pd.DataFrame([{'preg': 1,
                                'plas': 85,
                                'pres': 66,
                                'skin': 29,
                                'insu': 0,
                                'mass': 26.6,
                                'pedi': 0.351,
                                'age': 31}])
    
    model, normalized_data = normalize_data_minmax(input_data)
    normalized_data = pd.DataFrame(normalized_data, columns=input_data.columns)
    diabetes_model = load(open("./result_ft/model_diabetes.pkl", 'rb'))
    classes = diabetes_model.classes_
    diabetes_proba = diabetes_model.predict_proba(normalized_data)
    print(f'A classificação retornou como {diabetes_model.predict(normalized_data)[0]} para diabetes!')
    print('Distribuição Probabilistica:')
    print(f'Classe {classes[0]}: {diabetes_proba[0][0]}')
    print(f'Classe {classes[1]}: {diabetes_proba[0][1]}')
