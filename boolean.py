import pandas as pd
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')

# Obtém a lista de stop words em inglês
stop_words = set(stopwords.words('english'))


def remove_stop_words(text):
    words = text.split()
    filtered_words = [word for word in words if word.lower() not in stop_words]
    filtered_text = ' '.join(filtered_words)
    return filtered_text


def build_inverted_index(documents):
    inverted_index = {}
    for doc_id, doc_content in enumerate(documents):
        for term in remove_stop_words(doc_content).split():
            if term not in inverted_index:
                inverted_index[term] = set()
            inverted_index[term].add(doc_id)
    return inverted_index


def process_query(query, inverted_index, df, columns):
    query_terms = remove_stop_words(query).split()
    result_set = set()

    if len(query_terms) > 0:
        result_set = inverted_index.get(query_terms[0], set())

    for term in query_terms[1:]:
        if term in inverted_index:
            result_set = result_set.intersection(inverted_index[term])

    return df.iloc[list(result_set)]


def ranking(filePath, columns, query):
    # Leitura dos documentos a partir do arquivo CSV usando Pandas
    df = pd.read_csv(filePath)

    pd.set_option('display.max_colwidth', None)
    pd.set_option('display.expand_frame_repr', False)

    # Solicitar ao usuário as colunas a serem consultadas
    columns = columns.split()

    # Construção do índice invertido
    documents = df[columns].fillna('').apply(
        lambda row: ' '.join(row), axis=1).tolist()

    inverted_index = build_inverted_index(documents)

    # Processamento da consulta
    result_df = process_query(query, inverted_index, df, columns)

    # Exibição dos resultados
    print(result_df["Title"])
