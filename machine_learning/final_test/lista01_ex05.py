# Atividade final do módulo - EX05
# Autor: Newmar Wegner, Paulo, Kleberson
# Date: 23/05/2021

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
np.set_printoptions(suppress=True)

# Function to find optimal group clusters
def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = len(wcss), wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(numerator/denominator)
    
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
    plt.savefig('./result_ft/Kmeans_inertial_plot')
    plt.show()
    
    return distortions


if __name__ == '__main__':
    df = pd.read_csv('../datasets/fertility.txt')
    ## extraindo variavéis explicativas
    x = np.array(df.iloc[:, 0:9])
    ## identificando numero ótimo de clusters e clusterizando
    wcss = kmeans_inertial(x, 11)
    opt = optimal_number_of_clusters(wcss)
    # gerando o modelo com o numero ideal e salvando em uma variável
    kmeansmodel = KMeans(n_clusters=opt, random_state=1).fit(x)
