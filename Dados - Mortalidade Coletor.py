import requests
import csv

def pegarCsv(n1, primeira_vez=False):
    url = f"https://apidadosabertos.saude.gov.br/vigilancia-e-meio-ambiente/sistema-de-informacao-sobre-mortalidade?limit=1000&offset={n1*1000}"
    resposta = requests.get(url)
    dados = resposta.json()
    registros = dados.get("sim", [])

    if registros:
        modo = 'w' if primeira_vez else 'a'

        with open('dados.csv', modo, newline='', encoding='utf-8') as arquivo_csv:
            campos = registros[0].keys()
            escritor = csv.DictWriter(arquivo_csv, fieldnames=campos)

            if primeira_vez:
                escritor.writeheader()  # Só escreve o cabeçalho uma vez
            escritor.writerows(registros)
        print(f"Página {n1+1} salva com sucesso.")
    else:
        print(f"Nenhum dado encontrado na página {n1+1}.")

# Loop para baixar e salvar os dados paginados
for n in range(0, 2399):
    pegarCsv(n, primeira_vez=(n == 0))