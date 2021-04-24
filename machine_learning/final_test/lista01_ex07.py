# Atividade final do módulo - EX06
# Autor: Newmar Wegner, Paulo, Kleberson
# Date: 23/05/2021

# Desenvolver um classificador e validação cruzada com o arquivo BreastCancer
from pickle import dump

from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn import preprocessing, metrics
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split, cross_validate
from collections import Counter
import numpy as np


# pd.set_option('display.max_rows', 500)
# pd.set_option('display.max_columns', 500)
# pd.set_option('display.width', 1000)


# Function to normalize data considering Min Max Scaler sklearn


def normalize_data_minmax(df):
    normalizer = preprocessing.MinMaxScaler()
    
    return normalizer.fit(df), normalizer.fit_transform(df)


# Function to normalize dummy data
def normalize_data_dummies(df):
    return pd.get_dummies(df, prefix=df.columns)


def balance(attributes, classes):
    oversample = SMOTE()
    X, Y = oversample.fit_resample(attributes, classes)
    
    return X, Y


if __name__ == '__main__':
    # Open file and divide in attributes and classes
    df = pd.read_csv('../datasets/breastcancer.csv', sep=',')
    # ['age', 'menopause', 'tumor-size', 'inv-nodes', 'node-caps', 'deg-malig',
    #    'breast', 'breast-quad', 'irradiat', 'Class']
    numericals = pd.DataFrame(df['deg-malig'])
    categories = df.drop(columns=['deg-malig', 'Class'])
    class_column = df['Class']
    
    # normalizing numerical and dummies data
    model_norm, norm_num = normalize_data_minmax(numericals)
    norm_dummies = normalize_data_dummies(categories)
    df_normalized = pd.DataFrame(norm_num, columns=['deg-malig']).join(norm_dummies, how='left')
    df_normalized = df_normalized.join(pd.DataFrame(class_column, columns=['Class']), how='left')
    
    # Dividing attributes from class
    attributes = df_normalized.iloc[:, :-1]
    classes = df_normalized['Class']
    # print(classes.unique()) # show classes in column class
    # Balance dataset
    X, Y = balance(attributes, classes)
    
    # Prepare train and test dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3)
    
    # Get Random Forest model
    rf_model = RandomForestClassifier().fit(X_train, Y_train)
    
    # Predict Test values
    predicted = rf_model.predict(X_test)
    
    # Model accuracy
    print(f'Model Accuracy: {metrics.accuracy_score(Y_test, predicted)}')
    
    # Cross Validate
    cross_resuls = cross_validate(rf_model, X_train, Y_train, cv=10, return_train_score=True)
    print(f"Mean Cross validate Test Score: {cross_resuls['test_score'].mean()}")
    
    # Show confusion matrix and classification report considering classes
    print(confusion_matrix(Y_test, predicted))
    print(classification_report(Y_test, predicted, target_names=['no-recurrence-events', 'recurrence-events']))

    # Save Model
    dump(rf_model, open('./result_ft/rf_breast_cancer.pkl','wb'))
