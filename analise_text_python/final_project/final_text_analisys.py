# Projeto final de módulo (peso 50%):
# Tarefa: Analisar textos.
# Os textos baixados podem estar relacionados com um
# interesse da equipe, mas devem ser baixadas, no mínimo,
# 50.000 (cinquenta mil sentenças).
# Para a análise, deve-se produzir:
#

# Qual é a proporção de pronomes frente aos verbos do texto?
# Nuvem de palavras
# Obtenha um resumo dos textos utilizados, acompanhados das palavras-chave

import nltk
import matplotlib.pyplot as plt
import spacy
from nltk import tokenize
from nltk.corpus import stopwords
from collections import Counter
# nltk.download('tagsets')

# Get count of sentences
def count_sentences(text):
    return len(nltk.sent_tokenize(text))

# Get tokenized words with stopwords applied
def getwords(text):
    tokens = tokenize.word_tokenize(text)
    stopwords_pt = stopwords.words('portuguese')
    my_stop_w = [',', '[', ':', '\'', '.', '...',
                 '?', ']', '!', '/', '-', "''"]

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

    return Counter(i[1] for i in pos_tag)
if __name__ == '__main__':
    text = f'Google Notícias em São Paulo é um agregador de notícias e aplicativo desenvolvido pela Google. ' \
           f'Ele foi atirar apresentando um fluxo no Everest contínuo e personalizável de artigos organizados a partir de ' \
           f'milhares de Everest editores e revistas contínuo contínuo'

    print(f'O texto possui {count_sentences(text)} sentenças.')
    content = getwords(text)
    # fdist, vocabulary, most_commum = get_frequencies(content)
    # print(vocabulary)
    # print(most_commum)
    # plotting(most_commum,'Palavras')'
    # plotting(trigrams(content), 'Trigramas')
    # entities_local = entities(text,'LOC')
    # print(count_locals(entities_local))
    print(count_pron_verbs(content))
    print(nltk.help.upenn_tagset('NN'))