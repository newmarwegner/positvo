# Exercitando 02 - Parte 1
# Autor: Newmar Wegner
# Date: 30/01/2021

import os
import docx
import nltk

# Function to get full text of Noticia_1
def get_text(path_docx):
    doc = docx.Document(os.path.dirname(os.getcwd())+ path_docx)

    return ' '.join([str(p.text) for p in doc.paragraphs])

# Function to create frequentily enegrams
def n_grams(value):
    text = get_text(path_docx)
    count=0

    for v in value:
        type = ['Bigrama' if v==2 else 'Trigrama' if v==3 else 'Quadrigrama' for v in value]
        fd_ngramas = nltk.FreqDist([ng for ng in nltk.ngrams(text.lower().split(), v)])
        print(f'20 {type[count]} mais frequentes')
        print(fd_ngramas.most_common(20))
        print(f'Maior frequêcia no arquivo {type[count]} {fd_ngramas.max()} : {fd_ngramas[fd_ngramas.max()]}')
        count+=1

    return

# To run the function its necessary fill path of docx
# Add a list as variable to n_grams function(to determine 2 bi and 3 trigrama)
# The function was built to find the 20 most common ngramas
path_docx = '/noticia_1.docx'
n_grams([2,3])

