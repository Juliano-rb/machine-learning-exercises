import pandas as pd # learn more: https://python.org/pypi/pandas
import numpy as np
from time import time
import yellowbrick.cluster as ybc
from sklearn.cluster import KMeans


def evaluate(original_data, infer_labels):
    print("--Starting evaluation...")
    acertos = 0;
    total = len(infer_labels)

    for x in range(total):
        if original_data[x][0] == infer_labels[x]:
            acertos+=1

    print("--Taxa de acerto: " + str(acertos/total))


print("Loading dataset...", end=" ")
dataset_name = "wine.csv"
original = pd.read_csv(dataset_name, header=0)

# ARRAY usado para se conferir comparando as classes originais com as previstas
dataset_labeled = original.values

# Elimina a coluna que diz a classe das instâncias para o treinamento
df = original.drop(['class'], axis=1)

dataset = list(df.values)

print("done")

kmeans = KMeans(n_clusters=3, random_state=0)
predict_labels = kmeans.fit_predict(dataset)

# testas todas as instâncias verificando se a classe real delas é igual à classe agrupada pelo algoritmo
evaluate(dataset_labeled, predict_labels)

print("Exibindo Silhueta")
# Instantiate the clustering model and visualizer
#model = MiniBatchKMeans(6)
visualizer = ybc.SilhouetteVisualizer(kmeans)

visualizer.fit(np.asarray(dataset)) # Fit the training data to the visualizer
visualizer.poof() # Draw/show/poof the data

visualizer = ybc.KElbowVisualizer(kmeans, k=(2,7))

visualizer.fit(np.asarray(dataset)) # Fit the training data to the visualizer
visualizer.poof() # Draw/show/poof the data