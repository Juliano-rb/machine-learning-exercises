import pandas as pd
import yellowbrick.cluster
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D
import somoclu
import pandas as pd # learn more: https://python.org/pypi/pandas


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

n_rows, n_columns = 10, 10
som = somoclu.Somoclu(n_columns, n_rows, compactsupport=False)

print( np.float32(dataset[0]) )

som.train( np.asarray(dataset) )
# som.view_component_planes()


labels=range(3)
#som.view_umatrix(bestmatches=True )
som.view_activation_map(data_index=0)
som.view_activation_map(data_index=90)
som.view_activation_map(data_index=175)
# testas todas as instâncias verificando se a classe real delas é igual à classe agrupada pelo algoritmo
# evaluate(dataset_labeled, predict_labels)
