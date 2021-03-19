# Exercitando 03 - Parte 2
# Autor: Newmar Wegner
# Date:  05/02/2021


import nltk
import os
from nltk.corpus import machado
import pandas as pd

nltk.download('stopwords')


def gen_pos(arq, start_paragraph):
    list_text = machado.sents(arq)[start_paragraph:]
    words = []
    to_clean = ['*', 'A', 'O', 'D', 'E', '\x93', '\x97', ';', ',', '[', ':', '\'', '.', '...',
                '?', ']', '!', '/', '-', '(', ')']  # a option could be use string.punctuation
    stopwords = nltk.corpus.stopwords.words('portuguese')

    for paragraph in list_text:
        words.extend(paragraph)
    
    cleaned_words = [cw for cw in words if cw not in to_clean and cw not in stopwords]

    return nltk.pos_tag(nltk.word_tokenize(' '.join(cleaned_words)))

def gen_ner(pos_tag):
    entities = []
    labels = []
    chuncks = nltk.ne_chunk(pos_tag, binary=True)

    for chunck in chuncks:
        if hasattr(chunck, 'label'):
            entities.append(' '.join(c[0] for c in chunck))
            labels.append(chunck.label())

    return list(set(zip(entities, labels)))

def to_csv(dataframe, output_name):
    return dataframe.to_csv(f'{os.path.dirname(os.getcwd())}/parte_2/{output_name}.csv')

def commum_vocabulary(vocabularies):
    vocabularies_set = []
    for i in vocabularies:
        vocabularies_set.append(set(i))

    return vocabularies_set[0].intersection(*vocabularies_set)

def response_vocabulary():
    print(f'{len(commum_vocabulary(vocabularies))} ocorrências comuns registradas.')
    for i in commum_vocabulary(vocabularies):
        print(i)

    return

def frequency_most_commom(pos_tag, n_most_common):
    freqdist = nltk.FreqDist(pos_tag)

    return freqdist.most_common(n_most_common)

def count_entities(arq):
    pass

def gen_ner_without_set(pos_tag):
    entities = []
    labels = []
    chuncks = nltk.ne_chunk(pos_tag, binary=True)

    for chunck in chuncks:
        if hasattr(chunck, 'label'):
            entities.append(' '.join(c[0] for c in chunck))
            labels.append(chunck.label())

    return list(zip(entities, labels))


files = {"Dom Casmurro": ['romance/marm08.txt', 15],
         "O Alienista": ['contos/macn003.txt', 7]}

if __name__ == '__main__':
    vocabularies = []
    ## 1.
    print(machado.readme())
    ## 2.
    for item in files.items():
        ## 2.a
        print(f'Generating POS from book {item[0]} a partir do paragrafo {item[1][1]}')
        pos_tag = gen_pos(item[1][0], item[1][1])
        vocabularies.append(pos_tag)
        to_csv(pd.DataFrame(pos_tag), f'Newmar_POS_{item[0]}')
        ## 2.b
        print(f'Generating NER from book {item[0]} a partir do paragrafo {item[1][1]}')
        ner = gen_ner(pos_tag)
        to_csv(pd.DataFrame(ner), f'Newmar_NER_{item[0]}')
        ## 2.c
        print(pd.DataFrame(pos_tag).value_counts())
        print(pd.DataFrame(ner).value_counts())
        ## 2.d procedimento já ajustado na função gen_pos(arq,start_paragraph)
        ## 2.f
        print(frequency_most_commom(pos_tag, 10))
        ## 2.h
        print(pd.DataFrame(gen_ner_without_set(pos_tag)).value_counts())
        print('####################################################3')
    ## 2.e
    print('Vocabulários comuns:')
    response_vocabulary()
    ## 2.g
    print(len(commum_vocabulary(vocabularies)))
