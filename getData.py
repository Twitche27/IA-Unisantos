import requests
import json
import csv
import os

def getData(num):
    if num * 1000 >= 2399001:
        return

    key = ""
    secret = ""
    offset = num * 1000  # Para garantir que cada página pegue dados diferentes
    num += 1
    url = f'https://apidadosabertos.saude.gov.br/vigilancia-e-meio-ambiente/sistema-de-informacao-sobre-mortalidade?limit=1000&offset={offset}'

    headers = {"Content-Type": "application/json"}

    response = requests.get(url, headers=headers, auth=(key, secret))

    if response.status_code == 200 and response.json != None :
        data = response.json()

        print("Chaves principais:", data.keys())

        for key, value in data.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):


                file_exists = os.path.isfile("data.csv")

                with open("data.csv", mode="a", newline='', encoding="utf-8") as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=value[0].keys())

                    # Escreve cabeçalho apenas se o arquivo ainda não existir
                    if not file_exists or os.stat("data.csv").st_size == 0:
                        writer.writeheader()

                    writer.writerows(value)

                print(f"index {offset} - {len(value)} registros salvos com sucesso em data.csv")
                break
        else:
            print("Nenhuma lista de dicionários encontrada na estrutura do JSON.")

        getData(num)
    else:
        print(f"Erro na requisição: {response.status_code} - {response.text}")



getData(742000)