# Projeto final de módulo (peso 50%)
# Autor: Newmar Wegner, Paulo Gamero, Kleberson Nascimento
# Date: 15/03/2021

import os
import re
import nltk
import matplotlib.pyplot as plt
import spacy
from PyPDF2 import PdfFileReader
from gensim.summarization import summarize, keywords
from nltk import tokenize
from nltk.corpus import stopwords
from collections import Counter
from wordcloud import WordCloud

# nltk.download('tagsets')


# Get text from pdf book
def read_book():
    arq = open(os.path.dirname(os.getcwd())+'/final_project/sensing.pdf','rb')
    read_pdf = PdfFileReader(arq)
    text = []
    for i in range(21,read_pdf.getNumPages()-1):
        text.append(read_pdf.getPage(i).extractText())
    return '\n'.join(text)

# Get count of sentences
def count_sentences(text):
    return len(nltk.sent_tokenize(text))

# Get tokenized words with stopwords applied
def getwords(text):
    
    tokens = tokenize.word_tokenize(text.lower())
    stopwords_pt = stopwords.words('portuguese')
    regular = re.compile(r'(?<!\S)[A-Za-z]+(?!\S)|(?<!\S)[A-Za-z]+(?=:(?!\S))')
    tokens = filter(regular.match,tokens)
    
    my_stop_w = ['=', ',', '[', ':', '\'', '.', '...','x','b',
                 '?', ']', '!', '/', '-', "''",';','(',')']

    return [p for p in tokens if ((p not in stopwords_pt) and (p not in my_stop_w))]

# Get vocabulary of clean text
def get_frequencies(text):
    fdist = nltk.FreqDist(text)
    vocabulary = fdist.keys()
    most_commum = fdist.most_common(5)

    return fdist, vocabulary, most_commum

# Get Plot
def plotting(most_commum, text):
    most_commum_tri = []
    if text != 'Palavras':
        for i in most_commum:
            most_commum_tri.append([' '.join(i[0]), i[1]])

        plt.title(f'{text}: Frequência das 5 mais relevantes')
        plt.xticks(rotation=90)
        plt.gcf().subplots_adjust(bottom=0.50)
        plt.bar(*zip(*most_commum_tri))

    else:
        plt.title(f'{text}: Frequência das 5 mais relevantes')
        plt.bar(*zip(*most_commum))

    return plt.show()

# Get trigrams
def trigrams(content):
    trigams_list = []

    for p in nltk.ngrams(content, 3):
        trigams_list.append(p)

    fd_ngramas = nltk.FreqDist(trigams_list)

    return fd_ngramas.most_common(5)

# Get entities with filter name entity applied (eg: 'LOC')
def entities(text, filter):
    nlp = spacy.load('pt_core_news_sm')
    doc = nlp(text)
    classified = []
    for ent in doc.ents:
        classified.append([ent.text, ent.start_char, ent.end_char, ent.label_])

    return [i for i in classified if i[-1]==filter]

# Count locals
def count_locals(entities_local):
    locals =[]

    for i in entities_local:
        locals.append(i[0])

    return Counter(locals)

# Get Counts of pronoum and verbs in text
def count_pron_verbs(content):
    pos_tag = nltk.pos_tag(content)
    class_words = Counter(i[1] for i in pos_tag)

    pron_verbs = [['VB',0],['NN',0]]
    for i in class_words.items():
        if i[0].startswith('NN'):
            pron_verbs[1][1] += i[1]
        elif i[0].startswith('VB'):
            pron_verbs[0][1] += i[1]
    
    return pron_verbs

# Get wordcloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, relative_scaling=1.0, stopwords={'to', 'of', 'us'}).generate(text)
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.savefig('WordCloud.png')

    return plt.show()

# Get keywords
def get_keys(filter_words):

    return keywords(filter_words, ratio=0.05, pos_filter=('NN'), words=5)


# Create summarize with keywords
def summ_key(text):
    
    return summarize(text, ratio=0.15), get_keys(' '.join(getwords(text)))


if __name__ == '__main__':
    '''Texto utilizado para teste. Se utilizado é necessário comentar a linha 156'''
    # text = f'A pele, é o maior órgão do corpo humano, reveste e delimita o organismo. Reflete condições físicas e ' \
    #        f'psicológicas, como saúde, idade, diferenças étnicas e culturais. Desempenha importantes funções como a ' \
    #        f'proteção, excreção, termorregulação e percepções sensoriais. É composta por duas camadas distintas: a ' \
    #        f'epiderme,a camada mais superficial, constituída por um epitélio estratificado pavimentoso queratinizado, ' \
    #        f'e a derme, constituída por diversos tipos celulares, fibras colágenas e elásticas mergulhadas em uma ' \
    #        f'matriz extracelular onde também se situam os vasos e nervos; e, abaixo da pele encontra-se a hipoderme, ' \
    #        f'onde predomina o tecido adiposo, que une os órgãos subjacentes. Para restabelecer a integridade funcional,' \
    #        f' inicia-se um processo complexo Desempenha para a cicatrização da ferida e sua capacidade de reparação é muito ' \
    #        f'importante para a sobrevivência do indivíduo. Adiabetes influencia em diferentes etapas da cicatrização ' \
    #        f'de feridas, incluindo hemostasia e inflamação, deposição da matriz e angiogênese. No final do século XIX, ' \
    #        f'foi aplicada Arquivos do MUDI. A própolis éuma substância resinosa produzida por abelhas, a partir das ' \
    #        f'plantas como botões florais, exsudatos, brotos e pólen, que são modificadas pelas ações das secreções ' \
    #        f'salivares e enzimáticas, secretadas pelo metabolismo glandular desses insetos, imunoestimulatória ' \
    #        f'. Os flavonóides são os principais compostos com atividade farmacológica encontradosna própolis. ' \
    #        f'Atuam no processo de reparação tecidual, agem como antioxidantes, combatendo os radicais livres'
    
    
    text = read_book()
    # Contagem de sentenças
    print(f'O texto possui {count_sentences(text)} sentenças.')
    
    # Vocabulario e frequência de palavras relevantes
    content = getwords(text)
    fdist, vocabulary, most_commum = get_frequencies(content)
    print(vocabulary)
    print(most_commum)
    # Grafico das palavras mais relevantes e trigramas relevantes
    plotting(most_commum,'Palavras')
    plotting(trigrams(content), 'Trigramas')
    
    # Entidades LOC e count dessas entidades
    entities_local = entities(text,'LOC')
    print(entities_local)
    print(count_locals(entities_local))
    
    # Count de pronome e verbos no texto
    print(count_pron_verbs(content))
    
    # WorldCloud
    generate_wordcloud(text)
    
    # Resumo e palavras chaves
    summ_ , keys = summ_key(text)
    print(summ_)
    print(keys)
