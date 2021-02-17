# Exercitando 01 - Parte 2
# Autor: Newmar Wegner
# Date:  05/02/2021
import nltk
import os
from nltk.corpus import machado
from nltk.text import Text
# nltk.download('rslp')
# nltk.download('averaged_perceptron_tagger')
# romance/marm08.txt: Dom Casmurro (1899)
# contos/macn003.txt alienista

# # 1. Reading machado corpus
# # nltk.download('machado')
# print(machado.readme())
# # print(machado.fileids())
#
# # 2.a
# # ## Dom Casmurro
# arq = "romance/marm08.txt"
# doc = [paragraph for paragraph in machado.words(arq)]
# #
# # # Tokenizing
# words = nltk.word_tokenize(' '.join(str(p) for p in doc))
# # # Part of speach tagging
# pos_tags = nltk.pos_tag(words)
#
# # # Save post tags in txt file
# with open(os.path.dirname(os.getcwd())+'/parte_2/Newmar_POS_DomCasmurro.txt','w') as output:
#     for i in pos_tags:
#         output.write(str(i) + '\n')

## O alienista
arq = "contos/macn003.txt"
words = machado.words(arq)
print(' '.join(words))
# print(doc[3])
# # Tokenizing
# words = nltk.word_tokenize(' '.join(str(p) for p in doc))
#
# # Part of speach tagging
# pos_tags = nltk.pos_tag(words)
#
# # Save post tags in txt file
# with open(os.path.dirname(os.getcwd())+'/parte_2/Newmar_POS_DomCasmurro.txt','w') as output:
#     for i in pos_tags:
#         output.write(str(i) + '\n')



# paragrafo_stems = []
# #
# for paragrafo in paragrafos:
#   paragraph_aux = []
#   for sentence in paragrafo:
#     paragraph_aux.append([nltk.RSLPStemmer().stem(token) for token in sentence])
#   paragrafo_stems.append(paragraph_aux)
#
# print(paragrafos)
# print(paragrafo_stems)
