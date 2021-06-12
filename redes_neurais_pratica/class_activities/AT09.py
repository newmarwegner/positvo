# Atividade em aula - AT09
# Autor: Newmar Wegner
# Date: 11/06/2021

'''
Atividade – Conjunto de dados de Vozes
https://www.kaggle.com/primaryobjects/voicegender
•	Montar a parametrização;
•	Identificar no final o gênero da voz;
•	Comparar a acurácia obtida no site;
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from keras import models, layers
from tensorflow.keras.utils import to_categorical

pd.set_option('display.max_rows', 50)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 100)

# Organize train dataset
train = pd.read_csv('./data/voice.csv')

# Get number of unique values
print(train.label.unique())

# Create Train and test dataset
train.insert(0, 'identificador', range(0, 0 + len(train)))
train_set = train.sample(frac=0.7)
test_set = train[train.identificador.isin(train_set.identificador) == False]
# # Labels to categorical
label_train = np.array(train_set.iloc[:, -1:].values.ravel())
label_encoder = LabelEncoder()
train_setlabel = label_encoder.fit_transform(label_train)
train_label = to_categorical(train_setlabel)

label_test = np.array(test_set.iloc[:, -1:].values.ravel())
label_encoder = LabelEncoder()
test_setlabel = label_encoder.fit_transform(label_test)
test_label = to_categorical(test_setlabel)

# Exclude columns  that not will be use
train_set = train_set.iloc[:, 1:-1]
test_set = test_set.iloc[:, 1:-1]

# Convert to tensorflow datatype
train_set = tf.convert_to_tensor(train_set, dtype=tf.int64)
test_set = tf.convert_to_tensor(test_set, dtype=tf.int64)

print(test_set.shape)
print(train_set.shape)
print(test_label.shape)
print(train_label.shape)
print(train_set)

# Create model
network = models.Sequential()
network.add(layers.Dense(500, activation='relu', input_shape=(20,)))
network.add(layers.Dense(500, activation='relu'))
network.add(layers.Dense(2, activation='softmax'))
network.summary()
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Fit model
history = network.fit(train_set,
                      train_label,
                      batch_size=168,
                      epochs=1000,
                      validation_data=(test_set, test_label))

test_loss, test_acc = network.evaluate(test_set, test_label)
print('Test acurácia: ', test_acc)

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

