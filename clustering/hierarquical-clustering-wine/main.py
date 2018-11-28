import pandas as pd

import numpy as np
from time import time
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as hierarchy
import yellowbrick.cluster as ybc

def evaluate(original_data, infer):
    print("--Starting evaluation...")
    categories = list()
    acertos = 0;
    total = len(infer)
    #print("Total = " + str(total))

    for x in range(total):
        #print("Data:" + str(original_data[x][0]) + " label: " + str(infer[x]+1))
        if original_data[x][0] == infer[x]+1:
            acertos+=1

    print("--Taxa de acerto: " + str(acertos/total))

print("Loading dataset...", end=" ")
dataset_name = "wine.csv"

original = pd.read_csv(dataset_name, header=0)

labeled = original.values

df = original.drop(['class'], axis=1)

dataset = df.values

print("done")

# ward minimizes the variance of the clusters being merged.
# average uses the average of the distances of each observation of the two sets.
# complete or maximum linkage uses the maximum distances between all observations of the two sets.

'''
Carregamos os dados com pandas.
2 Usamos KMeans de sklearn.cluster
3 Elimina a coluna que diz a classe das instˆancias para o
treinamento.
4 Para cada ligação diferente, 'ward', 'average' e 'complete', cria-se um modelo definindo o número de clusters, e a função de distância.
5 Gera-se o dendograma para cada ligação diferente.
6 Apos isso:
1 Fazemos teste com todas as instancias verificando se a classe real
delas ´e igual `a classe agrupada pelo algoritmo.
'''
for linkage in ('ward', 'average', 'complete'):
    clustering = AgglomerativeClustering(affinity = 'euclidean', linkage=linkage, n_clusters=3)
    t0 = time()

    result = clustering.fit_predict(dataset)
    print("%s : %.2fs" % (linkage, time() - t0))
    print("--n_leaves: " + str(clustering.n_leaves_))

    evaluate(labeled, result)

    Z = hierarchy.linkage(dataset, linkage)

    plt.figure(figsize=(10, 10))
    plt.title('Dendrogram for ' + linkage + " linkage")
    plt.xlabel('sample index')
    plt.ylabel('distance')
    hierarchy.dendrogram(
        Z,
        truncate_mode='lastp',  # show only the last p merged clusters
        p=10,  # show only the last p merged clusters
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True  # to get a distribution impression in truncated branches
    )

    plt.show()

'''
    plt.figure(figsize=(10, 10))
    plt.title('Dendrogram for ' + linkage + " linkage")
    plt.xlabel('sample index')
    plt.ylabel('distance')
    hierarchy.dendrogram(
        Z,
        #truncate_mode='lastp',  # show only the last p merged clusters
        #p=3,  # show only the last p merged clusters
        leaf_rotation=90.,
        leaf_font_size=12.,
        show_contracted=True  # to get a distribution impression in truncated branches
    )

    plt.show()
'''