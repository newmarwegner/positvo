from nltk.corpus import machado, stopwords
import nltk
# nltk.download('machado')
print(machado.readme())

#2.a Categorias presentes no corpus
print(machado.categories())

#2.b
print(machado.fileids())

#2.c
arq = 'romance/marm05.txt'
palavras = machado.words([arq])
print(palavras)

#2.d
fdist = nltk.FreqDist(palavras)
for p in ['olhos','estado']:
    print(f'Arquivo {arq} e frequência da palavra {p} {fdist[p]}')

#2.e
print(len(palavras))

#2.f
print(len(fdist.keys()))

#2.g
print(fdist.keys())

#2.h
print(fdist.most_common(15))

#2.i
tabulate = fdist.tabulate(15)
print(tabulate)

#2.j
fdist.plot(15)

#2.k
stopword = stopwords.words('portuguese')
minha_stop_w = [',','[',':', '\'','.','...',
                    '?',']', '!','/','-',';','\x97',
                '(',')','\x94.','\x93','\x92',]
texto_limpo = []
for p in palavras:
    if p not in stopword and (p not in minha_stop_w):
        texto_limpo.append(p)

print(texto_limpo)

#2.l
n_trigramas= []
for ng in nltk.ngrams(texto_limpo, 3):
    n_trigramas.append(ng)

fd_ngramas = nltk.FreqDist(n_trigramas)
print(fd_ngramas.most_common())

#2.m
bigramas_olhos = []
alvo = 'olhos'
for p in nltk.ngrams(texto_limpo, 2):
    print(p)
    if alvo in p:
        bigramas_olhos.append(p)
fd_ngramas = nltk.FreqDist(bigramas_olhos)

print(fd_ngramas.most_common(15))
print(len(fd_ngramas.most_common(15)))

#2.n
fd_ngramas.plot(15)