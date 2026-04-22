import pandas as pd
import pickle

#carrega os modelos
normalizador = pickle.load(open("normalizador_boston.pkl", 'rb'))
cluster_boston = pickle.load(open("cluster_model_final.pkl", 'rb'))

colunas = [
    'CRIM', 
    'ZN', 
    'INDUS', 
    'CHAS', 
    'NOX', 
    'RM', 
    'AGE', 
    'DIS', 
    'RAD', 
    'TAX', 
    'PTRATIO', 
    'B', 
    'LSTAT', 
    'MEDV'
]

#cria um novo imovel
novo_imovel = [
    [0.15, 
    0.0, 
    5.0, 
    0, 
    0.5, 
    6.2, 
    50.0, 
    4.0, 
    3, 
    290.0, 
    18.0, 
    390.0, 
    10.0, 
    23.5]
]

#converte pra um dataframe do pandas
novo_imovel = pd.DataFrame(novo_imovel, columns=colunas)

# print(novo_imovel[['CRIM', 'RM', 'TAX', 'MEDV']])

#Normalizar os dados do imóvel
novo_imovel_norm_array = normalizador.transform(novo_imovel)

#manda o formato normalizado pro dataframe
novo_imovel_norm = pd.DataFrame(novo_imovel_norm_array, columns=colunas)

#inferencia
cluster_predict = cluster_boston.predict(novo_imovel_norm)

print(f"O imóvel pertence ao cluster {cluster_predict[0]}")   

