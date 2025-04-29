import os
import pandas as pd

caminho_pasta = os.path.expanduser("~/Downloads")
lista_dfs = []

for ano in range(2014, 2025):
    nome_arquivo = f"despesas_subfuncao {ano}.csv"
    caminho_arquivo = os.path.join(caminho_pasta, nome_arquivo)

    if os.path.exists(caminho_arquivo):
        print(f"Lendo arquivo corretamente: {nome_arquivo}")

        # Aqui está a chave para funcionar com BOM:
        df = pd.read_csv(caminho_arquivo, sep=';', encoding='utf-8-sig')
        df['Ano'] = ano
        lista_dfs.append(df)
    else:
        print(f"Arquivo não encontrado: {nome_arquivo}")

# Junta todos os dataframes
df_consolidado = pd.concat(lista_dfs, ignore_index=True)

# Exporta em UTF-8 com BOM também (bom para Excel)
df_consolidado.to_csv(os.path.join(caminho_pasta, "despesas_subfuncao_consolidado.csv"), index=False, encoding='utf-8-sig')

print("Tudo certo! Acentuação corrigida e arquivos unificados.")
