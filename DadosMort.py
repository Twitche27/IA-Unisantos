import pandas as pd

# Caminho para o CSV
caminho_csv = 'dados_mortalidade_combinados.csv'

# # Inicializa os contadores
# total_validos = 0
# total_dois = 0
# total_nove = 0
# total_vazios = 0
# total_nao_vazios = 0

primeiro_chunk = True

# # Lê o CSV em blocos
for chunk in pd.read_csv(caminho_csv, dtype=str, chunksize=100000):

    # Mantém os NaN e limpa os valores válidos
    chunk['ASSISTMED'] = chunk['ASSISTMED'].apply(
        lambda x: str(x).strip().replace('.0', '') if pd.notna(x) else x)

    # Filtra as linhas onde 'ASSISTMED' é igual a '1'
    filtrado = chunk[chunk['ASSISTMED'] == '1']

    # Salva o DataFrame filtrado em um novo arquivo CSV
    filtrado.to_csv('dados_mortalidade_filtrados.csv', mode='w' if primeiro_chunk else 'a', index=False, header=primeiro_chunk)

    primeiro_chunk = False

#     # Contagens
#     total_validos += (chunk['ASSISTMED'] == '1').sum()
#     total_dois += (chunk['ASSISTMED'] == '2').sum()
#     total_nove += (chunk['ASSISTMED'] == '9').sum()
#     total_nao_vazios += chunk['ASSISTMED'].notna().sum()
#     total_vazios += chunk['ASSISTMED'].isna().sum()

# # Exibe os resultados
# print(f'Total de registros não vazios: {total_nao_vazios} ({(total_nao_vazios / (total_nao_vazios+total_vazios) * 100):.2f}%)')
# print(f'Total de valores válidos (1): {total_validos} ({(total_validos / total_nao_vazios * 100):.2f}%)')
# print(f'Total de valores dois (2): {total_dois} ({(total_dois / total_nao_vazios * 100):.2f}%)')
# print(f'Total de valores ignorados (9): {total_nove} ({(total_nove / total_nao_vazios * 100):.2f}%)')
# print(f'Total de valores vazios: {total_vazios} ({(total_vazios / (total_nao_vazios+total_vazios) * 100):.2f}%)')
