# Atividade Final do Módulo
# Autor: Newmar Wegner, Paulo Gamero, Kleberson Nascimento
# Date: 25/06/2021

import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

qdf = pd.read_excel('perguntas.xlsx', sheet_name='Planilha1', usecols="A,B")
adf = pd.read_excel('perguntas.xlsx', sheet_name='Planilha2', usecols="A,B")


max_len = 15
max_words = 1000

tokenizer = Tokenizer(num_words=max_words)

tokenizer.fit_on_texts(qdf.question)
sequences = tokenizer.texts_to_sequences(qdf.question)

x_train = pad_sequences(sequences, maxlen=max_len)
y_train = to_categorical(qdf.answer_id)
model = Sequential()
model.add(Embedding(1000, 8, input_length=max_len))
model.add(Flatten())
model.add(Dense(adf.answer_id.max()+1, activation='softmax'))

model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['acc'])
model.summary()
history = model.fit(x_train, y_train, epochs=300, batch_size=22, validation_split=0.2)

# prediction = model(x_train)
# print(prediction)
# print(list(qdf.answer_id))
# print(list(np.argmax(prediction, axis=1)))

while True:
    sentence = input("Em que posso ajudar?: ").lower()
    sentence = tokenizer.texts_to_sequences([sentence.lower()])
    sentence = pad_sequences(sentence, maxlen=max_len)
    prediction = model(sentence)
    category = np.argmax(prediction, axis=1)
    answer = adf.query('answer_id=='+str(category)).to_numpy()
    print(answer)
    print("NewRobot answer:"+answer[0][1])
