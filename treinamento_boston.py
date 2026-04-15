import pandas as pd

caminho_arquivo = "HousingData.csv"
dados = pd.read_csv(caminho_arquivo)

#checando a quantidade de NaN 
print("\n\n\t\tAntes")
print(dados.isnull().sum())

#preenche o NaN com a média da respectiva coluna
dados = dados.fillna(dados.mean())

#checando a quantidade de NaN
print("\n\n\t\tDepois")
print(dados.isnull().sum())
print("\n\t\tFirst few Lines")
print(dados.head())
