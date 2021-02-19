# Exercitando 01 - Parte 2
# Autor: Newmar Wegner
# Date:  05/02/2021
import nltk
import os
from nltk.corpus import machado
import pandas as pd
# nltk.download('words')
# nltk.download('maxent_ne_chunker')
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
# 2.a
# ## Dom Casmurro
arq = "romance/marm08.txt"
doc = [paragraph for paragraph in machado.paras(arq)[6:]]
lista_text = []
for d in doc:
    lista_text.extend(d[0])

# # # Tokenizing
words = nltk.word_tokenize(' '.join(lista_text))
# # # Part of speach tagging
pos_tags = nltk.pos_tag(words)

# saving the postags file
with open(os.path.dirname(os.getcwd())+'/parte_2/Newmar_POS_DomCasmurro.txt','w') as output:
    for i in pos_tags:
        output.write(str(i) + '\n')

# # 2.b Save the entities
# Named entity recognition
chuncks = nltk.ne_chunk(pos_tags, binary=True)

# get entities inside of text
entities = []
labels = []
for chunck in chuncks:
    if hasattr(chunck,'label'):
        entities.append(' '.join(c[0] for c in chunck))
        labels.append(chunck.label())

# organizing list of entities and labels
entities_labels = list(set(zip(entities,labels)))

# Save ner in txt file
with open(os.path.dirname(os.getcwd())+'/parte_2/Newmar_NER_DomCasmurro.txt','w') as output:
    for i in entities_labels:
        output.write(str(i) + '\n')

df_entities = pd.read_csv(os.path.dirname(os.getcwd())+'/parte_2/Newmar_NER_DomCasmurro.txt')
df_entities.columns=['entidades','rotulos']
print(df_entities['entidades'].value_counts())
# df_entities.columns=['Entidades','Rotulos']
# print(df_entities['Entidades'].value_counts(ascending=False))
## O alienista
# arq = "contos/macn003.txt"
#
# words = machado.words(arq)
# print(' '.join(words))
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
