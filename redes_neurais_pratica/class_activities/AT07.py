# Atividade em aula - AT07
# Autor: Newmar Wegner
# Date: 11/06/2021

'''
Atividade - Clinica
https://www.kaggle.com/andrewmvd/heart-failure-clinical-data?select=heart_failure_clinical_records_dataset.csv
•	Criar um modelo para que tenha a maior acurácia possível;
o	Parametrizar o tamanho da rede;
o	Epocas;
'''

import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from keras import models, layers
from keras.regularizers import l2
from tensorflow.keras.utils import to_categorical

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Organize train dataset
train = pd.read_csv('./data/clinical.csv')

# Create Train and test dataset
train.insert(0, 'identificador', range(0, 0 + len(train)))
train_set = train.sample(frac=0.7)

test_set = train[train.identificador.isin(train_set.identificador) == False]

# Labels to categorical
train_label = to_categorical(train_set.DEATH_EVENT)
test_label = to_categorical(test_set.DEATH_EVENT)
#
# Exclude columns  that not will be use
train_set = train_set.iloc[:, 1:-1]
test_set = test_set.iloc[:, 1:-1]

# print(train_set)
# Convert to tensorflow datatype
train_set = tf.convert_to_tensor(train_set, dtype=tf.int64)
test_set = tf.convert_to_tensor(test_set, dtype=tf.int64)

# Create model
network = models.Sequential()
network.add(layers.Dense(14, activation='relu', input_shape=(12,), activity_regularizer=l2(0.001)))
network.add(layers.Dropout(0.4))
network.add(layers.Dense(14, activation='relu',  activity_regularizer=l2(0.001)))
network.add(layers.Dense(2, activation='softmax'))
network.summary()

network.compile(optimizer='adam',  loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model
history = network.fit(train_set,
                      train_label,
                      batch_size=10,
                      epochs=500,
                      validation_data=(test_set, test_label))

test_loss, test_acc = network.evaluate(test_set, test_label)
print('Test acurácia: ', test_acc)


'''
Resultados

Com uma camada oculta de 200 conexões
batch_size=128,
epochs=2000,
Test acurácia:  0.699999988079071


Com duas camadas ocultas [100,100]
batch_size=100,
epochs=4000,
Test acurácia:  0.7111111283302307

'''
##### Plots
history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

epochs = range(1, len(history_dict['loss']) + 1)

plt.plot(epochs, loss_values, 'bo', label='Training Loss')
plt.plot(epochs, val_loss_values, 'b', label='Validation Loss')
plt.title("Training and Validation Loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

acc_values = history_dict['accuracy']
val_acc_values = history_dict['val_accuracy']

plt.plot(epochs, acc_values, 'bo', label='Training Acc')
plt.plot(epochs, val_acc_values, 'b', label='Validation Acc')
plt.title("Training and Validation Accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()
