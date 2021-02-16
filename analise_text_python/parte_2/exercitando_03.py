# Exercitando 01 - Parte 2
# Autor: Newmar Wegner
# Date:  05/02/2021

from nltk.corpus import machado, stopwords
import nltk
# nltk.download('machado')
# print(machado.readme())
# print(machado.fileids())





arq = "romance/marm05.txt"
paragrafos = [paragraph for paragraph in machado.paras(arq)[17:19]]
print(paragrafos)

# paragrafo_stems = []
#
# for paragrafo in paragrafos:
#   paragraph_aux = []
#   for sentence in paragrafo:
#     paragraph_aux.append([RSLPStemmer().stem(token) for token in sentence])
#   paragrafo_stems.append(paragraph_aux)
#
# print(paragrafos)
# print(paragrafo_stems)
