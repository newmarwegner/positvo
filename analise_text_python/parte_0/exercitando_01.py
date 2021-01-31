# Exercitando 01 - Parte 0
# Autor: Newmar Wegner
# Date: 30/01/2021

## Creating a string variable
str = 'Ainda que falasse as línguas dos homens e falasse a língua dos anjos, sem amor eu nada seria.'

## Printing each character of string
for c in str:
    print(c)

## Split string to list
str_to_list = str.split(' ')
print(str_to_list)

## Showing count about how many words has on list
print(f'A lista possui {len(str_to_list)} palavras')

## Printing each word of stringI
for word in str_to_list:
    print(word)

## Change the value: "dos homens" to "do mundo
subs_str = str.replace('dos homens', 'do mundo')
print(subs_str)

## Printing slice 21:30 of string
print(str[21:30])

## Printing 15 last characters
print(str[-15:])

## Saving result in a txt file
f = open('saidaDoc.txt','w', encoding='utf8')
f.write(str[-15:])
f.close()


