# Atividade Final do Módulo
# Autor: Newmar Wegner, Paulo Gamero, Kleberson Nascimento
# Date: 25/06/2021
'''
Evidência 01 - mnist_vanilla (AT03)
Evidência 02 – mnist_vanilla_cnn (AT04)
Evidência 03 – CatDog (AT05)
Evidência 04 – Titanic (AT06)
Evidência 05 – Heart (AT07)
Evidência 06 – Wine Quality (AT08)
Evidência 07 – Gender Voice (AT09)
Evidência 08 – Orange - Titanic
Evidência 09 – Orange - Heart
Evidência 10 – Orange - Wine Quality
Evidência 11 – Orange - Gender Voice
Evidencia 12 - chatbox
'''

import os
import pandas as pd
import numpy as np
import speech_recognition as sr
from spellchecker import SpellChecker
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Embedding, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

def datasets():
    qdf = pd.read_excel('perguntas.xlsx', sheet_name='Planilha1', usecols="A,B")
    adf = pd.read_excel('perguntas.xlsx', sheet_name='Planilha2', usecols="A,B")
    qdf['question'] = qdf['question'].str.lower()
    
    return qdf, adf

def token(qdf):
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(qdf.question)
    sequences = tokenizer.texts_to_sequences(qdf.question)
    
    return tokenizer, sequences

def train(sequences, max_len, qdf):
    x_train = pad_sequences(sequences, maxlen=max_len)
    y_train = to_categorical(qdf.answer_id)
    
    return x_train, y_train

def create_and_fit_model():
    model = Sequential()
    model.add(Embedding(1000, 8, input_length=max_len))
    model.add(Flatten())
    model.add(Dropout(0.5))
    model.add(Dense(150))
    model.add(Dense(adf.answer_id.max() + 1, activation='softmax'))
    opt = optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['acc'])
    model.summary()
    history = model.fit(x_train, y_train, epochs=2000, batch_size=len(x_train), validation_split=0.2)
    return model

def test_model(model, x_train, qdf):
    prediction = model(x_train)
    print(list(qdf.answer_id))
    print(list(np.argmax(prediction, axis=1)))

# Função para captura, reconhecimento de voz e resposta
def recognize_speech_from_mic(recognizer, microphone):
    transcription = ""
    with microphone as source:
        audio = recognizer.listen(source)
        try:
            transcription = recognizer.recognize_google(audio, language="pt-br")
        except sr.RequestError:
            print("API unavailable")
        except sr.UnknownValueError:
            print('Repita a pergunta')
    return transcription


if __name__ == '__main__':
    max_len = 50
    max_words = 1000
    qdf, adf = datasets()
    tokenizer, sequences = token(qdf)
    x_train, y_train = train(sequences, max_len, qdf)
    model = create_and_fit_model()
    test_model(model, x_train, qdf)
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    microphone.device_index = 21
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source)

    os.system("say 'Olá, sou o NewRobot, vamos começar?'")
    while True:
        sentence = recognize_speech_from_mic(recognizer, microphone)
        if sentence == "": continue
        print("Você: " + sentence)
        sentence = tokenizer.texts_to_sequences([sentence.lower()])
        sentence = pad_sequences(sentence, maxlen=max_len)
        prediction = model(sentence)
        category = np.argmax(prediction, axis=1)[0]
        answer = adf.query('answer_id==' + str(category)).to_numpy()
        print("NewRobot: " + answer[0][1])
        os.system("say " + answer[0][1])

"""
Notas:
# Listando os devices para ver o indice
import sounddevice as sd
print (sd.query_devices())

# Para utilização do spell checker
sp = SpellChecker(language="pt")
# Verificar ortografia e colocar em minúsculas
def spck(sentences):
    checked_q = []
    for sentence in sentences:
        q = ""
        for word in sentence.lower().split():
            q = q + " " + sp.correction(word)
        checked_q.append(q)
    return checked_q

# Converter perguntas para minúsculas
qdf.question = spck(qdf.question)


# sem comando por voz
# while True:
#     sentence = input("Em que posso ajudar?: ").lower()
#     sentence = tokenizer.texts_to_sequences([sentence.lower()])
#     sentence = pad_sequences(sentence, maxlen=max_len)
#     prediction = model(sentence)
#     category = np.argmax(prediction, axis=1)
#     answer = adf.query('answer_id==' + str(category)).to_numpy()
#     print("NewRobot answer:" + answer[0][1])

"""
