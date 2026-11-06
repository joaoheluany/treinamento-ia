#IMPORTANTE: Antes de começar, instale os pacotes em requirements.txt
#
#python = -m pip install -r requirements.txt
 
import pandas as pd
import joblib
from sklearn.neighbors import KNeighborsClassifier
 
 
#IMPORTANTE: Instruções para Repositório de Dados
# 1. Criar um documento no Google Sheet
# 2. Insira dados com estes cabeçalhos: faltas, nota_1, nota_2, resultado
# 3. Adicione de 10 a 15 linhas conforme o exemplo abaixo:
#    8,2.5,4.5,reprovados
#    2,5.0,7.0,aprovado
#    .... (adicionar até 15 linhas)
# 4. Ir em "Arquivo" -> "Compartilhar" -> "Publicar na web"
# 5. Selecionar "valores separados por vírgula (.csv)" e clicar em "Publicar"
# 6. COPIAR O LINK GERADO E COLAR ABAIXO:
URL_DADOS = "https://docs.google.com/spreadsheets/d/e/2PACX-1vRbA7U31WWgNWfKvmSq8MPIzCVd-nno0GD-LkykGao7m5bqbHPEdkS9x1JqpfeE-INoVmdjmgvtMSE9/pub?output=csv"
 
ARQUIVO_MODELO = 'modelo_desempenho.pkl'
 
def treinar_modelo():
    print(f"Baixando dados de {URL_DADOS} ...")
    # Convertendo dados do arquivo csv em estrutura de dados (dataframe)
    data = pd.read_csv(URL_DADOS)
    print("-- Dados Carregados ---")
    print(data.head())
 
    print("--- Preparando dados para o treino")
    features = ["faltas", "nota_1", "nota_2"]
    target = "resultado"
    #X recebe as características(feature) e Y recebe o rótulo (chamado de alvo ou target, nesse caso)
    X = data[features]
    Y = data[target]
 
    modelo = KNeighborsClassifier(n_neighbors=3)
 
    modelo.fit(X, Y)
    print(f"--- Modelo Treinamento Classes: {modelo.classes_} ---")
 
    #Salvando o modelo treinamento no arquivo cujo nome está armazenado na constante ARQUIVO_MODELO
    joblib.dump(modelo, ARQUIVO_MODELO)
    print(f" --- Modelo salvo com sucesso em '{ARQUIVO_MODELO }' ---")
 
if __name__ == "__main__":
    treinar_modelo()