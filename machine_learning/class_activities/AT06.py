# Atividade em aula - AT06
# Autor: Newmar Wegner
# Date: 20/03/2021

'''
ATIVIDADE
1. Obter o modelo de clusters a partir da base fertility
2. Identificar o numero ideal de clusters
3. Gerar o model e salvar considerando o numero ideal de clusters
'''

import os
import pandas as pd
from pickle import dump
from sklearn.cluster import KMeans
from AT02 import optimal_number_of_clusters, kmeans_inertial



if __name__ == '__main__':
    df = pd.read_csv(os.path.dirname(__file__) + '/../datasets/fertility.txt', sep=',').drop("Output", axis=1)
    wcss = kmeans_inertial(df, 101)
    opt = optimal_number_of_clusters(wcss)
    kmeansmodel = KMeans(n_clusters=opt, random_state=1).fit(df)
    dump(kmeansmodel, open('results/modelclusterfertility.pkl','wb'))

