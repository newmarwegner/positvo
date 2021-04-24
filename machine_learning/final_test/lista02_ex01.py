# Atividade final do módulo - Lista 02 - EX01
# Autor: Newmar Wegner, Paulo, Kleberson
# Date: 23/05/2021

import pandas as pd
from pickle import dump
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing, metrics, svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


# Function to normalize data considering Min Max Scaler sklearn
def normalize_data_minmax(df):
    normalizer = preprocessing.MinMaxScaler()

    return normalizer.fit(df), normalizer.fit_transform(df)


# function to create numerical dataset
def create_num(df, class_column):
    numerical = df.drop(columns=class_column, axis=1)

    return numerical


# Function to Balance dataset
def balance_data(final_data, class_column):
    x, y = final_data.iloc[:, :-1], final_data[class_column]
    oversample = SMOTE()

    return oversample.fit_resample(x, y)


# Function to save model
def save_model(model):
    return dump(model, open('./result_ft/model_diabetes.pkl', 'wb'))

def save_best_model(metrics):
    v, i = max((v, i) for i, v in enumerate(metrics))
   
    return save_model(rf_diabetes) if i == 0 else save_model(svm_diabetes)
    


if __name__ == '__main__':
    # Input parameters
    df = pd.read_csv('../datasets/diabetes.csv', sep=',')
    class_column = ['class',]

    # print(create_num(df, class_column))
    # Creating numerical dataset
    numerical = create_num(df, class_column)

    # Numerical dataset to normalize with minmax scaler
    model, num_normalize = normalize_data_minmax(numerical)
    num_normalize = pd.DataFrame(num_normalize, columns=numerical.columns)
    print(numerical.columns)
    # Saving model normalizer
    dump(model, open('./result_ft/MinMaxScaler_diabetes.pkl', 'wb'))

    # Saving final normalizer dataset
    final_data = num_normalize.join(df['class'], how='left')
    final_data.to_csv('./result_ft/diabetes_normalize.csv', index=False, sep=';')

    # Get balanced dataset
    x_oversample, y_oversample = balance_data(final_data, class_column)

    # Slice test and train data
    X_train, X_test, Y_train, Y_test = train_test_split(x_oversample, y_oversample, test_size=0.3)
    
    # Get model Random Forest
    rf_diabetes = RandomForestClassifier().fit(X_train, Y_train.values.ravel())
    predicted_rf = rf_diabetes.predict(X_test)
    
    # Get Model SVM
    svm_diabetes = svm.SVC().fit(X_train, Y_train)
    predicted_svm = svm_diabetes.predict(X_test)
    
    # Get metrics of models
    metrics = [round(metrics.accuracy_score(Y_test, predicted_rf),3),
               round(metrics.accuracy_score(Y_test, predicted_svm),3)]
    
    # Accuracy: compare y_predict with Y_test
    print(f'Accuracy Model to identify diabetes cases (rf): {metrics[0]}')
    
    # Accuracy: compare y_predict with Y_test
    print(f'Accuracy Model to identify diabetes cases (svm): {metrics[1]}')

    # Save the best model
    save_best_model(metrics)
