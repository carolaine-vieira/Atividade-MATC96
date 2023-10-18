import vector
import boolean

filePath = input("Digite o caminho do arquivo:") or "doc.csv"
columns = input("Digite quais colunas utilizar:") or "Abstract"
query = input("Consulta: ") or "fuzzy"

vector.ranking(filePath, columns, query)
# vector.ranking_words(filePath, columns)
boolean.ranking(filePath, columns, query)
