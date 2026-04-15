import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

caminho_arquivo = "HousingData.csv"
dados = pd.read_csv(caminho_arquivo)

#checando a quantidade de NaN 
# print("\n\n\t\tAntes")
# print(dados.isnull().sum())

#preenche o NaN com a média da respectiva coluna
dados = dados.fillna(dados.mean())

#checando a quantidade de NaN
# print("\n\n\t\tDepois")
# print(dados.isnull().sum())
# print("\n\t\tFirst few Lines")
# print(dados.head())

# Nota pra prova: Como todas as colunas são númericas, da pra passar o dataset inteiro pro normalizador
scaler = MinMaxScaler() 

# treina o normalizador
normalizador = scaler.fit(dados)

pickle.dump(normalizador, open("normalizador_boston.pkl", "wb")) 

#aplica a transformação
dados_norm_array = normalizador.transform(dados) #o normalizador vai devolver uma matriz 

dados_norm = pd.DataFrame(dados_norm_array, columns=dados.columns)

print("\t\tDados normalizados• •")
print(dados_norm.head())







