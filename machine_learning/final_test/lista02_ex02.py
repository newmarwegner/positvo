# Atividade final do módulo - Lista 02 - EX02
# Autor: Newmar Wegner, Paulo, Kleberson
# Date: 23/05/2021


# 2.3 Após obter o modelo, liste as características de cada grupo (listar os valores dos centroides), analise-as e descreva os grupos
# 2.4 Sua análise pode ser realizada como comentário do programa ou como relatório gerencial. Opte pela forma que mais convier à equipe.

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pickle import dump
from imblearn.over_sampling import SMOTE
from sklearn import preprocessing
from sklearn.cluster import KMeans
np.set_printoptions(suppress=True)

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


# Function to Balance dataset
def balance_data(final_data, class_column):
    x, y = final_data.iloc[:, :-1], final_data[class_column]
    oversample = SMOTE()
    
    return oversample.fit_resample(x, y)


# Function to find optimal group clusters
def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = len(wcss), wcss[len(wcss) - 1]
    
    distances = []
    for i in range(len(wcss)):
        x0 = i + 2
        y0 = wcss[i]
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(numerator / denominator)
    
    return distances.index(max(distances)) + 2


# Function to run clusters
def kmeans_inertial(x, steps):
    distortions = []
    for k in range(1, steps):
        kmeansmodel = KMeans(n_clusters=k, random_state=1).fit(x)
        distortions.append(kmeansmodel.inertia_)
    
    fig, ax = plt.subplots()
    ax.plot(range(1, steps), distortions)
    ax.set(xlabel='Clusters', ylabel='Distorção', title='Método Elbow-Inertial')
    plt.savefig('./result_ft/Kmeans_inertial_plot_hypothyroid')
    plt.show()
    
    return distortions

# Function to format input dataset and return classes, numerics and dummies
def format_dataset(dataset):
    dataset = dataset.drop(columns='TBG').dropna().reset_index(drop=True)
    numerics = dataset[['age', 'TSH', 'T3', 'TT4', 'T4U', 'FTI']]
    dummies = dataset.drop(list(numerics.columns)+['Class'], axis=1)
    
    return dataset['Class'], numerics, dummies

# Function to concat dataframes
def concat_df(numerics_norm, dummies_norm, classes, num_columns):
    df = pd.DataFrame(numerics_norm, columns=numerics.columns).join(pd.DataFrame(dummies_norm), how='left')
    
    return df.join(classes, how='left')


if __name__ == '__main__':
    dataset = pd.read_csv('../datasets/hypothyroid.csv', sep=';')
    
    # Exclude TBG column and some lines due has null values
    classes, numerics, dummies = format_dataset(dataset)
    
    # Normalizing datasets
    model_norm, numerics_norm = normalize_data_minmax(numerics)
    dummies_norm = normalize_data_dummies(dummies)
    
    # Concatenating dataframes
    df_normalized = concat_df(numerics_norm, dummies_norm, classes, numerics.columns)
    
    # When check out dr_normalize was verified the secondary_hypothyroid class with just one line/
    # This characteristics result in a error on SMOTE algorithm to balance, besides that this row was/
    # exclude from dataset
    print(df_normalized['Class'].value_counts()) # show the counts of classes
    df_normalized = df_normalized.loc[df_normalized['Class'] != 'secondary_hypothyroid']
    
    # Balance dataset
    df_balanced = balance_data(df_normalized, 'Class')
    
    # Run Kmeans  and get optimal number of clusters
    wcss = kmeans_inertial(df_balanced[0], 11)
    opt = optimal_number_of_clusters(wcss)
    
    # Save Model with optimal clusters
    kmeansmodel = KMeans(n_clusters=opt, random_state=1).fit(df_balanced[0])
    dump(kmeansmodel, open('./result_ft/modelcluster_hypothyroid.pkl','wb'))
    print(kmeansmodel.cluster_centers_)
