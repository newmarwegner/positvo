# Exercitando 01 - Parte 1
# Autor: Newmar Wegner
# Date: 30/01/2021

import os
from nltk.corpus import CategorizedPlaintextCorpusReader

# Create a token path and reader with .txt files in folder
def token_reader():
    token_path = os.path.dirname(os.getcwd())+'/Dados/tokens/'
    reader = CategorizedPlaintextCorpusReader(
        token_path,
        '.*.txt',
        cat_pattern = '(\w+)/*',
        encoding = 'iso8859-1'
    )
    return token_path, reader

# Print negative and positive content
def contents(list_contents):
    token_path, reader = token_reader()
    for content in list_contents:
        if content[1:4] =='neg':
            print('\nConteúdo negativo')
        else:
            print('\nConteúdo positivo')

        for p in reader.words(token_path + content):
            print(p, ' ', end=' ')

    return

list_contents = ['/neg/cv002_tok-3321.txt',
                 '/pos/cv003_tok-8338.txt']
contents(list_contents)
