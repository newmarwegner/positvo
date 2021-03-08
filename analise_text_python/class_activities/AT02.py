# Atividade em aula - AT02
# Autor: Newmar Wegner
# Date: 05/03/2021

# Needs install package pt_core_news_sm with command python -m spacy download pt_core_news_sm

import spacy
import matplotlib.pyplot as plt
import wikipedia
from gensim.summarization import keywords
from gensim.summarization.summarizer import summarize
from nltk import tokenize, sent_tokenize
from nltk.corpus import stopwords
from wordcloud import WordCloud

# Get token text, pos and dep
def token_spacy(doc):
    classified = []
    for token in doc:
        classified.append([token.text, token.pos_, token.dep_])

    return classified

# Get entities
def entities(doc):
    classified = []
    for ent in doc.ents:
        classified.append([ent.text, ent.start_char, ent.end_char, ent.label_])

    return classified

# Using Matplotlib
def plotting():
    x = [1, 2, 3, 4, 5, 6, 7, 8]
    y = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    plt.plot(x, y)

    return plt.show()

# Get wikipedia content
def wikicontent(lang, search):
    wikipedia.set_lang(lang)
    wikisearch = wikipedia.page(search)

    return wikisearch.content

# Get tokenized words with stopwords applied
def getwords(content):
    tokens = tokenize.word_tokenize(conteudo)
    stopwords_pt = stopwords.words('portuguese')

    return [p for p in tokens if not p in stopwords_pt]

# Get keywords
def get_keys(filter_words):

    return keywords(filter_words, ratio=0.05, pos_filter=('NN'), words=5)

# Get wordcloud
def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, relative_scaling=1.0, stopwords={'to', 'of', 'us'}).generate(text)
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.savefig('WordCloud.png')

    return plt.show()



if __name__ == '__main__':
    nlp = spacy.load('pt_core_news_sm')
    text = f'Google Notícias é um agregador de notícias e aplicativo desenvolvido pela Google. ' \
           f'Ele apresenta um fluxo contínuo e personalizável de artigos organizados a partir de ' \
           f'milhares de editores e revistas.' \

    doc = nlp(text)
    print(f"Resultando palavras e suas classes: \n {token_spacy(doc)}")
    print(f'Entidades obtidas: \n {entities(doc)}')

    ## Wikipedia analysis text
    # Create content
    conteudo = wikicontent('pt','Usina Hidrelétrica de Itaipu')
    # Create summarize
    summ_= summarize(conteudo, ratio=0.02)
    # Create tokenize content with stopwords applied
    filter_words = getwords(conteudo)
    # Create keywords
    keys_w = get_keys(' '.join(filter_words))
    # Create wordcloud
    text = 'O senhor é o meu pastor e rei, o senhor é rei'
    generate_wordcloud(text)
