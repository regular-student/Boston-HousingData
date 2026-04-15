import pickle
import pandas as pd

caminho_arquivo = "HousingData.csv"
dados = pd.read_csv(caminho_arquivo)
cluster_model = pickle.load(open("cluster_model_final.pkl", "rb"))

dados['Cluster'] = cluster_model.labels_

# agrupar os dados por cluster
descricao_clusters = dados.groupby('Cluster').mean()

# colunas essênciais pra entender um bairro
colunas_chave = ['CRIM', 'RM', 'TAX', 'MEDV']

print("\t•Perfil médio de cada cluster\n")
print(descricao_clusters[colunas_chave].round(2))

## salvando a tabela em CSV
descricao_clusters.to_csv("descricao_clusters_boston.csv")
