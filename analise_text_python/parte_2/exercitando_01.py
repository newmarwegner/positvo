import os
import docx
import nltk
from nltk import PorterStemmer, LancasterStemmer, word_tokenize
# nltk.download('rslp')

# Open document with docx
arq = os.path.dirname(os.getcwd())+'/Noticia_2.docx'
doc = docx.Document(arq)

# Create a paragraphs list
paragraph_list = []
for p in doc.paragraphs:
    paragraph_list.append(p.text)

## filter second and third paragraph
paragraph_analisys = paragraph_list[2:4]

## tokenizing paragraphs
tokens_second = word_tokenize(' '.join(paragraph_analisys))

## Porter stemmer
porter=PorterStemmer()
stems_porter=[]
for i in tokens_second:
    stems_porter.append(porter.stem(i))

## Lancaster stemmer
lancaster = LancasterStemmer()
stems_lancaster = []
for i in tokens_second:
    stems_lancaster.append(lancaster.stem(i))

## RSLP stemmer
rslp = nltk.RSLPStemmer()
stems_rslp = []
for i in tokens_second:
    stems_rslp.append(rslp.stem(i))

## Printing porter and lancaster list of second paragraph
print('Printing porter, lancaster and rslp list to compare')
print(stems_porter)
print(stems_lancaster)
print(stems_rslp)
