# Atividade em aula - AT06
# Autor: Newmar Wegner
# Date: 11/06/2021

'''
Atividade - Titanic
https://www.kaggle.com/hesh97/titanicdataset-traincsv
.	Criar um modelo para que tenha a maior acurácia possível;
.	Parametrizar o tamanho da rede;
.	Epocas;
'''
import pandas as pd
import tensorflow as tf
from keras import models, layers
from tensorflow.keras.utils import to_categorical

pd.set_option('display.max_rows', 100)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

# Organize train dataset
train = pd.read_csv('./data/train_titanic.csv')

# Convert categorical data male and female
train['Sex'] = train['Sex'].astype('category')
train['Sex'] = train['Sex'].cat.codes
train['Embarked'] = train['Embarked'].astype('category')
train['Embarked'] = train['Embarked'].cat.codes

# Drop unnecessary columns
train.drop(['Name', 'Ticket', 'Cabin'], axis=1, inplace=True)
train = train.dropna()

# Create Train and test dataset
train_set = train.sample(frac=0.7)
test_set = train[train.PassengerId.isin(train_set.PassengerId) == False]

# Labels to categorical
train_label = to_categorical(train_set.Survived)
test_label = to_categorical(test_set.Survived)

# Exclude columns  that not will be use
train_set = train_set.iloc[:, 2:]
test_set = test_set.iloc[:, 2:]

# Convert to tensorflow datatype
train_set = tf.convert_to_tensor(train_set, dtype=tf.int64)
test_set = tf.convert_to_tensor(test_set, dtype=tf.int64)

# Create model
network = models.Sequential()
network.add(layers.Dense(200, activation='relu', input_shape=(7,)))
network.add(layers.Dense(2, activation='softmax'))
network.summary()
network.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print(train_set.shape)
print('############################')
print(test_set.shape)
print('######################3')
print(train_label.shape)
print('############################')
print(test_label.shape)


# # Fit model
# history = network.fit(train_set,
#                       train_label,
#                       batch_size=128,
#                       epochs=2000,
#                       validation_data=(test_set, test_label))
#
# test_loss, test_acc = network.evaluate(test_set, test_label)
# print('Test acurácia: ', test_acc)
#

'''
Resultados

Com uma camada oculta de 200 conexões
batch_size=128,
epochs=2000,
Test acurácia:  0.8084112405776978

batch_size=60,
epochs=1000,
Test acurácia:  0.8130841255187988

'''
