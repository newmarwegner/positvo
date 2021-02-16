# Exercitando 02 - Parte 2
# Autor: Newmar Wegner
# Date: 16/02/2021

import os
import docx
import nltk
# nltk.download('tagsets')

# Open document with docx
arq = os.path.dirname(os.getcwd())+'/Noticia_1.docx'
doc = docx.Document(arq)

paragraph_list = []
for p in doc.paragraphs:
    paragraph_list.append(p.text)

# Tokenizing
words = nltk.word_tokenize(' '.join(paragraph_list))

# Part of speach tagging
pos_tags = nltk.pos_tag(words)

# Save post tags in txt file
with open(os.path.dirname(os.getcwd())+'/parte_2/Newmar_POS_Noticia1.txt','w') as output:
    for i in pos_tags:
        output.write(str(i) + '\n')

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
print(entities_labels)

# Save ner in txt file
with open(os.path.dirname(os.getcwd())+'/parte_2/Newmar_NER_Noticia1.txt','w') as output:
    for i in entities_labels:
        output.write(str(i) + '\n')
