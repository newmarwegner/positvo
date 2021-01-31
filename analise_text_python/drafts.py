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
'''