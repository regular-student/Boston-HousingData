import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import pickle

from scipy.spatial.distance import cdist
from sklearn.cluster import KMeans
import math
import numpy as np

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

# -> treina o normalizador
normalizador = scaler.fit(dados)

pickle.dump(normalizador, open("normalizador_boston.pkl", "wb")) 

#aplica a transformação
dados_norm_array = normalizador.transform(dados) #o normalizador vai devolver uma matriz 

dados_norm = pd.DataFrame(dados_norm_array, columns=dados.columns)

# print("\t\tDados normalizados• •")
# print(dados_norm.head())

## -> a ideia agora é descobrir o número ótimo de clusters

# testando de 1 a 45
K = range(1, dados.shape[0])
distortions = []

for i in K:
    cluster_model_temp = KMeans(n_clusters=i, random_state=42).fit(dados_norm)

    distortions.append(
        sum(
            np.min(
                cdist(dados_norm, cluster_model_temp.cluster_centers_, 'euclidean'), axis=1
            ) / dados_norm.shape[0]
        )
    )


# matemática pra achar o cotovelo da reta
x0 = K[0]
y0 = distortions[0]
xn = K[-1]
yn = distortions[-1]
distances = []

for i in range(len(distortions)):
    x = K[i]
    y = distortions[i]
    numerador = abs((yn-y0)*x - (xn-x0)*y + xn*y0 - yn*x0)
    denominador = math.sqrt((yn-y0)**2 + (xn-x0)**2)
    distances.append(numerador/denominador)

# número de clusters correspondente a maior distancia calculada
numero_clusters_otimo = K[distances.index(np.max(distances))]

# print(numero_clusters_otimo)

# -> treinar o modelo definitivo 
cluster_model_final = KMeans(
    n_clusters=numero_clusters_otimo,
    random_state=42
).fit(dados_norm)

# salvando o modelo treinado dessa vez
pickle.dump(cluster_model_final, open('cluster_model_final.pkl', 'wb'))
print("checkpoint: success!")
