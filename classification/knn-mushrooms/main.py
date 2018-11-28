##Todas as variaveis##
'''class, cap-shape, cap-surface, cap-color, bruises, odor, gill-attachment, gill-spacing,
   gill-size, gill-color, stalk-shape, stalk-root, stalk-surface-above-ring, stalk-surface-below-ring,
   stalk-color-above-ring, stalk-color-below-ring, veil-type, veil-color, ring-number, ring-type,
   spore-print-color,population,habitat

'''
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import simplejson as json
import math
from numbers import Number
#import seaborn as sns # data viz
#from subprocess import check_output
def load_data(csv_dataset):
    print("Carregando dataset..")
    df = pd.read_csv(csv_dataset)

    #Criando um array com os itens
    data_prepared = df.values.tolist()

    data_prepared = clean_data(data_prepared)

    return (data_prepared)

def clean_data(array_data):
    for x in array_data:
        if '?' in x:
            array_data.remove(x)
        if len(x) != 23:
            print("--tupla com tamanho diferente")

    return  array_data

def get_distance(instance1, instance2):
    dif = 0
    for x in range(1, len(instance1)):
        if instance1[x] != instance2[x]:
            dif+=1

    #distance = dif/len(instance1)

    return dif
#funcao key do metodo sorted é chamada sempre antes da comparação para efetuar alguma operação sobre o dado que está sendo
def key_func(item):
    return item["distance"]

def inferir(dataset, instancia, k):
    data_distance = list()

    for mush in dataset:
        distance = get_distance(mush, instancia)
        data_distance.append( {"mush":mush, "distance":distance} )

    data_distance = sorted(data_distance, key=key_func)

    data_distance=data_distance[:k]

    votos = {
        'e':0,
        'p':0
    }

    for x in data_distance:
        mush = x[ 'mush' ]
        votos[ mush[0] ]+=1

    if votos['e'] > votos['p']:
        return 'e'
    elif votos['p'] > votos['e']:
        return 'p'

def analise(data):
    edible = 0
    poisonous = 0

    #A categoria está na primeira posicao do array, ao contrario de outros
    for x in data:
        if(x[0] == 'e'):
            edible+=1
        elif(x[0] == 'p'):
            poisonous+=1

    print("Total comestível: " + str(edible) + "\nTotal Venenoso: " + str(poisonous))

data_prepared = load_data("mushrooms.csv")
analise(data_prepared)

train_data = data_prepared[0 : 5000]
test_data = data_prepared[5000 : len(data_prepared)]

print("Tamanho do dataset de treinamento: " + str(len(train_data)))
print("Tamanho do dataset de teste: " + str(len(test_data)))


ks = [1, 3, 5, 7, 11, 13, 15, 17, 19, 21]
filename = "Results.txt"
results = open(filename, 'w')

results.write("Resultados:\nK\tTaxa de acerto")
for k in ks:
    print("\nIniciando teste para " + "k = " + str(k))
    acertos = 0
    total = len(test_data)

    atual = 1
    for x in test_data:
        resposta = inferir(train_data, x, k)
        if resposta == x[0]:
            acertos+=1


        print("\r" + str( '{0:.2f}'.format( (atual/total) * 100) ) + "% concluido. Taxa de acerto: " + str('{0:.2f}'.format(acertos/atual)), end="")

        atual+=1

    print()
    print("Taxa de acerto para k = " + str(k) +  ": " + str(acertos/total))
    results.write("\n" + str(k) + "\t" + str(str(acertos/total)))

results.close()

