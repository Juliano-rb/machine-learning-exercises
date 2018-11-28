import pandas as pd
from sklearn.neural_network import MLPClassifier as MLP

# Valores possíveis para cada atributo:
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
COLUMN_NAMES=['class', 'cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor',
       'gill-attachment', 'gill-spacing', 'gill-size', 'gill-color',
       'stalk-shape', 'stalk-root', 'stalk-surface-above-ring',
       'stalk-surface-below-ring', 'stalk-color-above-ring',
       'stalk-color-below-ring', 'veil-type', 'veil-color', 'ring-number',
       'ring-type', 'spore-print-color', 'population', 'habitat']

COLUMN_NAMES.append(COLUMN_NAMES.pop(COLUMN_NAMES.index('class')))

def preprocessing_csv(file_name):
    print("Fazendo pre processamento no dataset...")
    df = pd.read_csv(file_name)

    # Embaralha as instancias do dataset
    df = df.sample(frac=1)
    # Ordena as colunas no dataset de acordo com a ordem definida em COLUMN_NAMES (Apenas foi colocada a coluna class para o fim
    df = df.reindex(columns=COLUMN_NAMES)
    count_dropped = 0
    for index, row in df.iterrows():
        # print(dir(row))
        # print(row._get_axis(0))

        i = 0
        for d in row.values:
            if d == "?":
                df = df.drop(index)
                count_dropped += 1
                break

            d = binary_category(d, row._get_axis(0)[i])

            df.loc[index][row._get_axis(0)[i]] = d
            i+=1
    print("Removidos " + str(count_dropped) + " instancias por falta de dados.")
    n_file_name = "mushrooms_preprocessed.csv"
    df.to_csv(n_file_name, index = False)

    return n_file_name

# converte o dado categorico representado por um caractere em uma categoria representada por um numero em binario
def binary_category(char_feature, column_name):
    column = attrs[column_name].index(char_feature)

    bin_cat = bin(column)[2:].zfill(4)

    return bin_cat

def load_data(csv_dataset):
    print("Carregando dataset...")
    df = pd.read_csv(csv_dataset, header=0)

    train = df[0:4000]
    test = df[4000:len(df.axes[0])]

    # 1. Assign the DataFrame's labels (the right-most column) to train_label.
    # 2. Delete (pop) the labels from the DataFrame.
    # 3. Assign the remainder of the DataFrame to train_features
    train_features, train_label = train, train.pop("class")

    test_features, test_label = test, test.pop("class")

    return ( train_features.values.tolist(), train_label.values.tolist() ), ( test_features.values.tolist(), test_label.values.tolist() )

def analise(data):
    edible = 0
    poisonous = 0
    df = pd.read_csv(data, header=0)

    #A categoria está na primeira posicao do array, ao contrario de outros
    for index, row in df.iterrows():
        x = list(row)

        if( int(x[-1]) == 0):
            edible+=1
        elif( int(x[-1]) == 1):
            poisonous+=1

    print("Total comestível: " + str(edible) + "\nTotal Venenoso: " + str(poisonous))

def test(layers_size, learning_rate, activation_func, epoch):
    clf = MLP(learning_rate_init=learning_rate, activation=activation_func, hidden_layer_sizes=layers_size, max_iter=epoch)

    clf.fit(train_f, train_l)

    score = clf.score(test_f, test_l)
    return score



# file_name = preprocessing_csv("mushrooms.csv")./
analise("mushrooms_preprocessed.csv")
(train_f, train_l), (test_f, test_l ) = load_data("mushrooms_preprocessed.csv")


print("Tamanho do conjunto de treinamento: " + str(len(train_f)) + "\nTamanho do conjunto de teste: " + str(len(test_f)))

base = [(20,20), 0.001, 'logistic',  10000]
qtd_tests = 10;
print("Modelo base: " + str(base))
results = open("Resultados.txt", 'a')

# Configurações para cada variação dos testes efetuados
hidden_layers = [ [3,3], [5,5], [10,10], [15,15], [20,20],[25,25], [30,30], [40,40] ]
learn_rates = [0.1,0.01,0.001,0.0001,0.00001,0.000001]
activation_funcs = [ 'identity', 'logistic', 'tanh', 'relu']
epochs = [1, 10, 20, 50, 100, 500, 1000, 2000]

# Execução com uma única camada escondida variando a quantidade de neurônios
hidden_layers_one_layer = [1,5,10,15,20,25,30]
print("\n\n---Iniciando teste Menos é mais...---")
results = open("Resultados.txt", 'a')
results.write('MENOS é mais\n')
tests_results = []
for h in hidden_layers_one_layer:
    print(str(h) + ": ", end="")
    amostras = []
    for x in range(qtd_tests):
        acuracy = test(h, base[1], base[2], base[3])
        print("teste " + str(x) + ": " + str(acuracy )+ ",", end="")
        amostras.append(acuracy)
    print()
    acuracy = ( sum(amostras)/(len(amostras)) )
    print("media com camadas escondidas: " + '{0:.2f}'.format(acuracy) + "% ", end="")
    results.write(str(h) + '\t' + str(acuracy * 100) + '\n')
    print()
#
# Abaixo estão as execuções feitas variando os atributos para a geração dos gráficos
#
'''
tests_results = []
for h in hidden_layers:
    print(str(h) + ": ", end="")
    amostras = []
    for x in range(qtd_tests):
        acuracy = test(h, base[1], base[2], base[3])
        print("teste " + str(x) + ": " + str(acuracy )+ ",", end="")
        amostras.append(acuracy)
    print()
    acuracy = ( sum(amostras)/(len(amostras)) )
    print("media:" + '{0:.2f}'.format(acuracy) + "% ", end="")
    results.write(str(h[0]) + '\t' + str(acuracy * 100) + '\n')
    print()
'''
'''
print("Variando taxas de aprendizado... ")
results.write("Taxas de aprendizado:\nLearning rate\tAcuracy\n")
tests_results = []
for l in learn_rates:
    print(str(l) + ": ", end="")
    amostras = []
    for x in range(qtd_tests):
        acuracy = test(base[0], l, base[2], base[3])
        print("teste " + str(x) + ": " + str(acuracy )+ ",", end="")
        amostras.append(acuracy)
    print()
    acuracy = ( sum(amostras)/(len(amostras)) )
    print("media:" + '{0:.2f}'.format(acuracy) + "% ", end="")
    results.write(str(l) + '\t' + str(acuracy * 100) + '\n')
    print()
'''
'''
print("Variando função de ativação... ")
results.write("Funções de ativação:\nFunção\tTaxa de acerto\n")
tests_results = []
for f in activation_funcs:
    print(str(f) + ": ", end="")
    amostras = []
    for x in range(qtd_tests):
        acuracy = test(base[0], base[1],f, base[3])
        print("teste " + str(x) + ": " + str(acuracy )+ ",", end="")
        amostras.append(acuracy)
    print()
    acuracy = ( sum(amostras)/(len(amostras)) )
    print("media:" + '{0:.2f}'.format(acuracy) + "% ", end="")
    results.write(str(f) + '\t' + str(acuracy * 100) + '\n')
    print()
'''
'''
print("Variando quantidade de épocas... ")
results.write("Épocas:\nÉpoca\tTaxa de acerto\n")
tests_results = []
for e in epochs:
    print(str(e) + ": ", end="")
    amostras = []
    for x in range(qtd_tests):
        acuracy = test(base[0], base[1],base[2], e)
        print("teste " + str(x) + ": " + str(acuracy )+ ",", end="")
        amostras.append(acuracy)
    print()
    acuracy = ( sum(amostras)/(len(amostras)) )
    print("media:" + '{0:.2f}'.format(acuracy) + "% ", end="")
    results.write(str(e) + ' ' + str(acuracy * 100) + '\n')
    print()
'''