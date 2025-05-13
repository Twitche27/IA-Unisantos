import pandas as pd
from collections import Counter

# Caminho do arquivo CSV grande
caminho_csv = 'dados_mortalidade_filtrados.csv'

# Inicializa contador
contador_mensal = Counter()

# Define o tamanho dos blocos de leitura
chunk_size = 100000

# Processa em blocos
for chunk in pd.read_csv(caminho_csv, chunksize=chunk_size, dtype=str):

    # Extrai os últimos 6 dígitos
    chunk['ULTIMOS_6'] = chunk['DTOBITO'].str[-6:]

    # Extrai mês e ano
    chunk['DATA'] = chunk['ULTIMOS_6'].str[2:] + '-' + chunk['ULTIMOS_6'].str[:2]

    # Conta ocorrências por mês
    counts = chunk['DATA'].value_counts()

    # Atualiza o contador total
    for data, count in counts.items():
        contador_mensal[data] += count

# Transforma o Counter em DataFrame
df_resultado = pd.DataFrame(contador_mensal.items(), columns=['DATA', 'MORTALIDADE - ÓBITOS'])

# Converte 'DATA' para datetime para ordenação cronológica
df_resultado['DATA_ORDENADA'] = pd.to_datetime(df_resultado['DATA'], format='%Y-%m')

# Ordena corretamente
df_resultado = df_resultado.sort_values('DATA_ORDENADA')

# Remove a coluna auxiliar
df_resultado = df_resultado.drop(columns='DATA_ORDENADA')

# Salva resultado
df_resultado.to_csv('mortalidade_por_mes.csv', index=False)
print("Arquivo 'mortalidade_por_mes.csv' salvo com sucesso e ordenado por data.")
