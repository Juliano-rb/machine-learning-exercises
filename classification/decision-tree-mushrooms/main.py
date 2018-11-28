import csv
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.tree import tree
from sklearn import tree as tree2
import pandas as pd

#carrega o dataset
data_set = pd.read_csv('mushrooms.csv')
''' 
    Referente à função de transformar em binario as categorias
'''
attrs = {
    'class': ['e', 'p'],
    'cap-shape': ['b','c','x','f','k','s'],
    'cap-surface': ['f','g','y','s'],
    'cap-color': ['n','b','c','g','r','p','u','e','w','y'],
    'bruises': ['t','f'],
    'odor': ['a','l','c','y','f','m','n','p','s'],
    'gill-attachment': ['a','d','f','n'],
    'gill-spacing': ['c','w','d'],
    'gill-size': ['b','n'],
    'gill-color': ['k','n','b','h','g','r','o','p','u','e','w','y'],
    'stalk-shape': ['e','t'],
    'stalk-root': ['b','c','u','e','z','r','?'],
    'stalk-surface-above-ring': ['f','y','k','s'],
    'stalk-surface-below-ring': ['f','y','k','s'],
    'stalk-color-above-ring': ['n','b','c','g','o','p','e','w','y'],
    'stalk-color-below-ring': ['n','b','c','g','o','p','e','w','y'],
    'veil-type': ['p','u'],
    'veil-color': ['n','o','w','y'],
    'ring-number': ['n','o','t'],
    'ring-type': ['c','e','f','l','n','p','s','z'],
    'spore-print-color': ['k','n','b','h','r','o','u','w','y'],
    'population': ['a','c','n','s','v','y'],
    'habitat': ['g','l','m','p','u','w','d']
}

# converte o dado categorico representado por um caractere em uma categoria representada por um numero em binario
def binary_category(char_feature, column_name):
    column = attrs[column_name].index(char_feature)
    bin_cat = bin(column)[2:].zfill(4)

    return bin_cat
'''
    FIm do referente à funcao de classificar binario
'''
#row contem os nomes dos atributos
feature_names = np.array(data_set.columns.values[1:])

# Carrega o conjunto de treino e as classes
data_set_x, data_set_y = [], []

# iterrows retorna um objeto que dar pra usar no for, dataframe normal não dá pelo q vi, retorna outra coisa,
# index, row recebe a tupla dos dados retornados por iterrows, por isso recebo os dois no for, pra ficarem separados na hr de usar
for index, row in data_set.iterrows():
    # todos elementos da row exeto a classe
    features = row[1: ]

    data_set_x.append( features )
    # label
    data_set_y.append( row[0] )  # A classe que diz se é comestível ou não

data_set_x = np.array(data_set_x)
data_set_y = np.array(data_set_y)
#Converte os datasets para a forma binaria
# array com features...
for i in range(len(data_set_x)):
    for j in range(len(data_set_x[i])):
        #print(data_set.axes[1])
        data_set_x[i][j] = binary_category(data_set_x[i][j], data_set.axes[1][j+1] )
# e array com classes (menos opcoes, usei apenas um if)
for i in range(len(data_set_y)):
    if data_set_y[i] == 'e':
        data_set_y[i] = 0
    elif data_set_y[i] == 'p':
        data_set_y[i] = 1

# Conjuntos de treino e de teste
x_train,x_test,y_train,y_test = train_test_split(data_set_x,data_set_y, test_size=0.25, random_state=33)

# Geração de resultados
'''
depths = [ 1, 2, 3, 4, 5, 10]

results = open("Resultados.txt", 'w')
results.write("produndidade maxima taxa de acerto\n")
for d in depths:
    clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=d, min_samples_leaf=5)
    clf = clf.fit(x_train, y_train)
    print("max_depth = " + str(d))
    score = clf.score(x_test, y_test)
    print(score)
    results.write(str(d) + " " + str(score*100) + "\n")
'''
clf = tree.DecisionTreeClassifier(criterion='entropy', max_depth=2, min_samples_leaf=5)
clf = clf.fit(x_train, y_train)

score = clf.score(x_test, y_test)
print(score)

tree2.export_graphviz(clf, out_file='tree.dot')
