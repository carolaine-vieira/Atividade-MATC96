import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Baixe as stopwords (apenas na primeira execução)
nltk.download('stopwords')
nltk.download('punkt')


def process_text(text):
    tokens = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)


def build_inverted_index(documents):
    inverted_index = {}
    document_titles = {}

    for doc_id, (title, doc_content) in enumerate(documents):
        doc_content = process_text(doc_content)
        document_titles[doc_id] = title
        for term in doc_content.split():
            if term not in inverted_index:
                inverted_index[term] = set()
            inverted_index[term].add(doc_id)

    return inverted_index, document_titles


def process_query(query, inverted_index):
    query = process_text(query)
    query_terms = query.split()
    result_set = set()

    if len(query_terms) > 0:
        result_set = inverted_index.get(query_terms[0], set())

    for term in query_terms[1:]:
        if term in inverted_index:
            result_set = result_set.intersection(inverted_index[term])

    return result_set


def ranking(filePath, columns, query):
    # Leitura dos documentos a partir do arquivo CSV usando Pandas
    df = pd.read_csv(filePath)

    columns_to_consult = columns.split(',')

    # Garantir que as colunas existem no DataFrame
    for col in columns_to_consult:
        if col not in df.columns:
            print(f"A coluna '{col}' não existe no DataFrame.")
            return

    # Selecionar os títulos e textos das colunas especificadas e processar com NLTK
    documents = [(title, abstract)
                 for title, abstract in df[columns_to_consult].values]

    documents = [(title, process_text(abstract))
                 for title, abstract in documents]

    # Construção do índice invertido e mapeamento de títulos
    inverted_index, document_titles = build_inverted_index(documents)

    # Processamento da consulta
    result_set = process_query(query, inverted_index)

    # Exibição dos resultados
    print("\nDocumentos relevantes para a consulta '{}':".format(query))
    for doc_id in result_set:
        print(f"Título: {document_titles[doc_id]}")
        print(f"Abstract: {documents[doc_id][1]}")
        print("="*30)
