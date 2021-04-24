# Atividade final do módulo - EX06 inferência
# Autor: Newmar Wegner, Paulo, Kleberson
# Date: 23/05/2021


import pandas as pd
from sklearn import preprocessing
from pickle import load
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

# Function to divide input in dummy_columns and numerical columns
def dummy_num(data_to_predict):
    dummy_columns = []
    numerical = []
    
    for i in data_to_predict:
        if isinstance(i, str):
            dummy_columns.append(i)
        else:
            numerical.append(i)
    dummy_columns = pd.DataFrame(dummy_columns).T
    dummy_columns.columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month',
                             'poutcome']
    
    return dummy_columns, pd.DataFrame(numerical)


if __name__ == '__main__':
    # Input parameters
    df = pd.read_csv('../datasets/bank.csv', sep=';')
    dummy_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']
    class_column = ['y', ]

    # Creating categorical and numerical dataset
    categorical, numerical = create_catnum(df, dummy_columns, class_column)

    # Get dummies to categorical dataset
    norm_categorical = normalize_data_dummies(categorical)

    # Join norm_categorical with norm_numerical dataset
    model_norm, num_normalize = normalize_data_minmax(numerical)
    num_normalize = pd.DataFrame(num_normalize, columns=numerical.columns)
    dados = num_normalize.join(norm_categorical, how='left')
    print(dados)

    # Load model and make predictions
    rf = load(open("./result_ft/model_bank.pkl",'rb'))
    print(rf.predict(dados))
