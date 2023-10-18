import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def ranking(filePath, columns, query):
    # Ler a planilha CSV
    df = pd.read_csv(filePath)

    # Separe os nomes das colunas usando a vírgula como delimitador
    nomes_colunas = columns.split(',')
    nomes_colunas = [nome.strip() for nome in nomes_colunas]
    docs = df[nomes_colunas[0]]
    for nome_coluna in nomes_colunas[1:]:
        docs = docs + ' ' + df[nome_coluna]

    # Inicializar o vetorizador TF-IDF
    vector = TfidfVectorizer(stop_words='english')

    # Transformar os documentos e a query em vetores TF-IDF
    matrix = vector.fit_transform(docs)
    vector_query = vector.transform([query])

    # Calcular a similaridade cosseno entre a query e os documentos
    similarity = cosine_similarity(vector_query, matrix)

    # Criar uma lista de tuplas (título do documento, pontuação de similaridade)
    ranqueamento = [(df.iloc[i]['Title'], similaridade)
                    for i, similaridade in enumerate(similarity[0])]

    # Ordenar a lista pelo valor de similaridade em ordem decrescente
    ranqueamento = sorted(ranqueamento, key=lambda x: x[1], reverse=True)

    # document = []

    # Iterar sobre o array do ranqueamento pra gerar o arquivo
    for i, (titulo, score) in enumerate(ranqueamento, start=1):
        # linha = {'Pontuação': score, 'Título': titulo}
        if (score > 0):
            print(score)
            print(titulo + "\n")
        # document.append(linha)

    # ranking_file = pd.DataFrame(document)
    # ranking_file.to_csv('ranking.csv')


def ranking_words(filePath, columns):
    # Ler a planilha CSV
    df = pd.read_csv(filePath)

    # Separe os nomes das colunas usando a vírgula como delimitador
    nomes_colunas = columns.split(',')
    nomes_colunas = [nome.strip() for nome in nomes_colunas]
    docs = df[nomes_colunas[0]]
    for nome_coluna in nomes_colunas[1:]:
        docs = docs + ' ' + df[nome_coluna]

    # Inicializar o vetorizador TF-IDF
    vector = TfidfVectorizer(stop_words='english')

    # Transformar os documentos e a query em vetores TF-IDF
    matrix = vector.fit_transform(docs)
    # Acesse as pontuações por palavra
    termos = vector.get_feature_names_out()
    pontuacoes_por_palavra = matrix.sum(axis=0).A1

    # Criar um dicionário com os termos e suas pontuações
    pontuacoes_dict = dict(zip(termos, pontuacoes_por_palavra))

    # Exemplo: Exibir as pontuações por palavra
    for termo, pontuacao in pontuacoes_dict.items():
        print(f"Termo: {termo} - Pontuação: {pontuacao}")
