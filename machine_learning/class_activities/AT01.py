# Atividade em aula - AT01
# Autor: Newmar Wegner
# Date: 19/03/2021

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

## function to find optimal value K
def optimal_number_of_clusters(wcss):
    x1, y1 = 2, wcss[0]
    x2, y2 = 20, wcss[len(wcss)-1]

    distances = []
    for i in range(len(wcss)):
        x0 = i+2
        y0 = wcss[i]
        numerator = abs((y2-y1)*x0 - (x2-x1)*y0 + x2*y1 - y2*x1)
        denominator = np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        distances.append(numerator/denominator)
    
    return distances.index(max(distances)) + 2

def kmeans_distance(x, steps):
    distortions = []
    for k in range(1, steps):
        kmeansmodel = KMeans(n_clusters=k).fit(x)
        distortions.append(
            sum(np.min(
                cdist(x, kmeansmodel.cluster_centers_, 'euclidean'), axis=1
            )) / x.shape[0])
    
    fig, ax = plt.subplots()
    ax.plot(range(1, steps), distortions)
    ax.set(xlabel='Clusters',ylabel='Distorção',title='Método Elbow-Dist')
    plt.savefig('results/Kmeans_distance_plot')
    plt.show()

    return distortions

if __name__ == '__main__':
    df = pd.read_csv(os.path.abspath(os.path.join(os.path.dirname( __file__ ),'..','datasets/fertility.txt')))
    ## extraindo variavéis explicativas
    x = np.array(df.iloc[:, 0:9])
    wcss = kmeans_distance(x, 11)
    print(optimal_number_of_clusters(wcss))
