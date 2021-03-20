'''
# Abrindo pdfs
import os
from PyPDF2 import PdfFileReader
path = os.getcwd()+ '/Dados/PrimeiroPDF.pdf'
cont_arquivo = open(path,'rb')
cont_pdf = PdfFileReader(cont_arquivo)
print(cont_pdf.getNumPages())

paragraph_list = []
for i in range(0, cont_pdf.getNumPages()):

    paragraph_list.append(cont_pdf.getPage(i).extractText())
print(paragraph_list)
print(''.join(paragraph_list))

exemplo 02 aula
from nltk import PorterStemmer


tokens = ['wait', 'waiting', 'waited', 'waiter', 'waitress']
porter = PorterStemmer()

for t in tokens:
    print(porter.stem(t))

import nltk
# nltk.download('punkt')
# nltk.download('rslp')
from nltk import word_tokenize, PorterStemmer, LancasterStemmer
from nltk import RSLPStemmer


texto = f'My name is Maximus Decimus Meridius, commander of the armies of the north, General of the Felix legions and loyal servant to the true emperor, Marcus Aurelius.' \
      f'Father to a murdered son, husband to a murdered wife. And I will have my vengeance, in this life or the next (Gladiator, the movie).'

# tokenizar
tokens = word_tokenize(texto)

# Instancia de objetos com os stemmers
porter= PorterStemmer()
lancaster = LancasterStemmer()
rslp = RSLPStemmer()

# Stemming para cada token
stem_text_porter = []
stem_text_lancaster = []

for palavra in tokens:
      stem_text_porter.append(porter.stem(palavra))
      stem_text_lancaster.append(lancaster.stem(palavra))
print('Porter Stemmer')
print(stem_text_porter)
print('Lancaster Stemmer')
print(stem_text_lancaster)

exemplo 03
texto = f'Meu nome é Maximus Decimus Meridius, comandante dos exércitos do norte, general das legiões de Félix e servo ' \
        f'leal ao verdadeiro imperador, Marcus Aurelius. Pai de um filho assassinado, marido de uma esposa assassinada. ' \
        f'E eu terei minha vingança, nesta vida ou na próxima (Gladiador, o filme).'

# tokenizar
tokens = word_tokenize(texto)

# Instancia de objetos com os stemmers
rslp = RSLPStemmer()

# Stemming para cada token
stem_text_rslp = []

for palavra in tokens:
    stem_text_rslp.append(rslp.stem(palavra))

print('Porter rslp')
print(stem_text_rslp)



# Exemplo 04
# pos - part of speach

# import nltk
# import pandas as pd
# import numpy as np
# from nltk import word_tokenize
# # nltk.download('maxent_ne_chunker')
# nltk.download('words')
# # nltk.download('averaged_perceptron_tagger')
# text_e = f'Apple acquires Zoom in China on wednesday 6th may 2020. This news has made Apple and Google stock jump by ' \
#          f'5% in the United States of America.'
#
# text_p = f'Apple adquire Zoom na China na quinta-feira 6 de maio de 2020. Essa notícia fez as ações da Apple e da G' \
#          f'oogle subirem 5% nos Estados Unidos.'
#
# tokens_e = word_tokenize(text_e)
# tokens_pt = word_tokenize(text_p)
#
# pos_tags = nltk.pos_tag(tokens_e)
#
# # criando chuncks
# chuncks = nltk.ne_chunk(pos_tags,binary=True)
# for c in chuncks:
#     print(c)
#
# # recuperar as entidades do texto
# entidades = []
# rotulos = []
#
# for c in chuncks:
#     if hasattr(c, 'label'):
#         entidades.append(' '.join(elemento[0] for elemento in c))
#         rotulos.append(c.label())
#
# print( '#######################')
# print('NER - resultado')
#
# entidades_com_rotulos = list(set(zip(entidades,rotulos)))
# print(entidades_com_rotulos)
#
#
# entitidades_df = pd.DataFrame(entidades_com_rotulos)
#
#
# entitidades_df.columns=['Entidades','Rotulos']
# print(entitidades_df)


## Processos com spaCy
## Utiliza redes neurais para processamento
# 1 - instalar o spacy pip installl
# 2 - baixar o modelo do idioma em questap
#
# import spacy
# nlp = spacy.load("pt_core_news_sm")
#
# texto = 'Apple adqure Zoom na China na quinta-feira 6 de maio de 2020. Essa notícia fez as ações.'
#
# doc = nlp(texto)
# print(doc.text)
# for token in doc:
#     print(token.text,token.pos_,token.dep_)
import os

from wordcloud import WordCloud
import matplotlib.pyplot as plt

def generate_wordcloud(text):
    wordcloud = WordCloud(width=800, height=400, relative_scaling=1.0, stopwords={'to', 'of', 'us'}).generate(text)
    plt.axis('off')
    plt.imshow(wordcloud)
    plt.savefig('WordCloud.png')

    return plt.show()

Exercicio para ainda serem realizados
Rever exercitando03_parte_2
Resumos e palavras-chave da página da Wikipedia ("Usina de Itaipu")
Nuvem de palavras da página da Wikipedia


 # Create wordcloud
arq = open('territorio','r')
text = arq.read()

generate_wordcloud(text)
'''

####################### Machine learning #############################################
# import os
#
# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.cluster import KMeans
# from sklearn import metrics
# from scipy.spatial.distance import euclidean, cdist
#
# iris = pd.read_csv(os.path.dirname(__file__) + '/machine_learning/datasets/iris.csv')
# x0 = iris.iloc[:, 0]
# x1 = iris.iloc[:, 1]
# x2 = iris.iloc[:, 2]
# x3 = iris.iloc[:, 3]
# y = iris.iloc[:, 4]
# ## extraindo variavéis explicativas
# x = np.array(iris.iloc[:, 0:4])
### maneira do professor
# x = np.array(list(zip(x0, x1,x2,x3))).reshape(len(x0),4)
# print(x)
# cluster = KMeans(n_clusters=3).fit(x)
# print(cluster.cluster_centers_)

# distortions = []
# for k in range(1, 11):
#     kmeansmodel = KMeans(n_clusters=k).fit(x)
#     distortions.append(
#         sum(np.min(
#             cdist(x, kmeansmodel.cluster_centers_, 'euclidean'), axis=1
#         )) / x.shape[0])
#
# fig, ax = plt.subplots()
# ax.plot(range(1, 11), distortions)
# ax.set(xlabel='Clusters',ylabel='Distorção',title='Método Elbow')
# plt.savefig('plot')
# plt.show()


####################### com inercia##############################################3


# distortions = []
# for k in range(1, 11):
#     kmeansmodel = KMeans(n_clusters=k).fit(x)
#     distortions.append(kmeansmodel.inertia_)
#
# fig, ax = plt.subplots()
# ax.plot(range(1, 11), distortions)
# ax.set(xlabel='Clusters',ylabel='Distorção',title='Método Inertial')
# plt.savefig('Inertial')
# plt.show()

####################### normalização #############################################3
import os
import pandas as pd
import sklearn
from sklearn import preprocessing

df = pd.read_csv(os.path.dirname(__file__) + '/machine_learning/datasets/dados_normalizar.csv', sep=';')
# print(df.head())

#1. obter o vetor de valores não numéricos
#1.1 Segmentar as colunas que são numéricas, para normalização quantitativa
df_num = df.drop(columns=['sexo'])
df_categories = df['sexo']
# print(df_num)
# print(df_categories)
#1.2 Obter o vetor numérico a partir dos dados que são numéricos
df_num.x = df_num.values
# print(df_num.x)
# 2 Normalização utilizando o método min max manualmente
#Z = (x - min(dados)/max(dados)-min(dados))
df_num_norm_minmax = (df_num - df_num.min())/(df_num.max()-df_num.min())
# print(df_num_norm)

# 2 Normalização utilizando o método média anual
# Z = (x - media(atributo)/ desvio padrão do atributo
df_num_norm_mean = (df_num - df_num.mean())/df_num.std()
# print(df_num_norm_mean)

# 3 Utilizando MinMaxScaler()
# 3.1 Criar um normalizador
normalizador = preprocessing.MinMaxScaler()
df_num_norm_model= normalizador.fit(df_num)
df_num_norm_scaler = normalizador.fit_transform(df_num)
print(df_num_norm_scaler)
