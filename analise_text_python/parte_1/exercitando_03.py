# Exercitando 03 - Parte 1
# Autor: Newmar Wegner
# Date: 30/01/2021

import nltk
from nltk.corpus import webtext, stopwords

# Function to analyze the frequency of words list considering archives
def frequency(filter, arq):
    for p in filter:
        print(f'Arquivo {arq}'
              f' e frequência da palavra {p}'
              f' {nltk.FreqDist(webtext.words(arq))[p]}')

    return webtext.words(arq)

# Function to clean characteres
def tokenize(words):
    stopword = stopwords.words('english')
    my_stop_w = [',','[',':', '\'','.','...',
                    '?',']', '!','/','-']

    return [p for p in words if ((p not in stopword) and (p not in my_stop_w))]

# Function to create bi-tri or quadrigramas considering value to most commom
def n_grams(words,value,common):
    clean_text = tokenize(words)
    type = ['Bigrama' if v == 2  else 'Trigrama' if v==3 else 'Quadrigrama' for v in value]

    count = 0
    for v in value:
        print(f'Lista dos {common} bigramas mais frequentes do texto')
        fd_ngramas = nltk.FreqDist([ng for ng in nltk.ngrams(clean_text, v)])
        print(f'{common} {type[count]} mais frequentes\n',
              f'{fd_ngramas.most_common(common)}\n'
              f'Maior frequêcia no arquivo {type[count]}'
              f' {fd_ngramas.max()} : {fd_ngramas[fd_ngramas.max()]}')
        count += 1

    return

def run(arq_list, filter_list,list_ngramas,common):
    for arq in arq_list:
        words = frequency(filter_list, arq)
        n_grams(words,list_ngramas,common)

    return

# Create a list with 15 frequentily bigramas with words 'the' and 'that'
run(['singles.txt','pirates.txt'], ['the','that'],
    [2,], 15)

print('=================++++++++++++++++================')

#Create a list with 20 frequentily bigramas with word 'life'
run(['singles.txt','pirates.txt'], ['life'],
    [4,], 20)

