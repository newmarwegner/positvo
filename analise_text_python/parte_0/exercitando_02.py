# Exercitando 02 - Parte 0
# Autor: Newmar Wegner
# Date: 30/01/2021

import os
import docx

# Open document with docx
arq = os.path.dirname(os.getcwd())+'/Dados/ROMANCE.docx'
doc = docx.Document(arq)

# Create a paragraphs list
paragraph_list = []
for p in doc.paragraphs:
    paragraph_list.append(p.text)

# Count paragraphs on doc
#print(f'O texto tem {len(paragraph_list)} parágrafos')

# Print first paragraph
print(paragraph_list[0])

# Print paragraph 3 and 6
print(paragraph_list[2], '\n',
     paragraph_list[5])

# Verify if "Machado" in document
if 'Machado' in paragraph_list:
    print('O termo existe')
else:
    print('O termo não existe')

## Create a full text with paragraphs on list
full_text = '\n'.join(paragraph_list)
print(full_text)

## Change Batista to João Batista in full_text
full_text_changed = full_text.replace('Batista','João Batista')
print(full_text_changed)
