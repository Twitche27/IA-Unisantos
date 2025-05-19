import pandas as pd
import os

# Caminho da pasta com os arquivos CSV
pasta_csv = 'GastosSaudePublica'

todos_dfs = []

# Percorre todos os arquivos da pasta
for nome_arquivo in os.listdir(pasta_csv):
    if nome_arquivo.endswith('.csv'):
        caminho_arquivo = os.path.join(pasta_csv, nome_arquivo)
        df = pd.read_csv(caminho_arquivo, sep=';')
        todos_dfs.append(df)

# Concatena todos os DataFrames
df_final = pd.concat(todos_dfs, ignore_index=True)

df_final = df_final.loc[:, ~df_final.columns.str.startswith('Unnamed')]

df_final = df_final.rename(columns={'Mês Ano': 'Data'})

# Converte a coluna "Data" para datetime
df_final['Data'] = pd.to_datetime(df_final['Data'], format='%m/%Y', errors='coerce')

# Ordena pela coluna "Data"
df_final = df_final.sort_values('Data').reset_index(drop=True)

# Converte de volta para o formato Ano-mes
df_final['Data'] = df_final['Data'].dt.strftime('%Y-%m')

# Salva em um novo CSV
df_final.to_csv('dados_unificados.csv', index=False)
