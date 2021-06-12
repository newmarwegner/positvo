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
import matplotlib.pyplot as plt
from sklearn import preprocessing
from keras import models, layers
from tensorflow.keras.utils import to_categorical
from keras.regularizers import l2
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

# # Normalizar dados
# min_max_scaler = preprocessing.MinMaxScaler()
# normalize_data = min_max_scaler.fit_transform(train_set)
# train_set = pd.DataFrame(normalize_data, columns=train_set.columns)
#
#
# normalize_data = min_max_scaler.fit_transform(test_set)
# test_set = pd.DataFrame(normalize_data, columns=test_set.columns)


# print(train_set)
#
# Convert to tensorflow datatype
train_set = tf.convert_to_tensor(train_set, dtype=tf.int64)
test_set = tf.convert_to_tensor(test_set, dtype=tf.int64)

# Create model
network = models.Sequential()
network.add(layers.Dense(30, activation='relu', input_shape=(7,), activity_regularizer=l2(0.001)))
network.add(layers.Dense(20, activation='relu', activity_regularizer=l2(0.001)))
network.add(layers.Dense(2, activation='softmax'))
network.summary()
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
network.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])

print(train_set.shape)
print('############################')
print(test_set.shape)
print('######################3')
print(train_label.shape)
print('############################')
print(test_label.shape)

# Fit model
history = network.fit(train_set,
                      train_label,
                      batch_size=100,
                      epochs=128,
                      validation_data=(test_set, test_label))

test_loss, test_acc = network.evaluate(test_set, test_label)
print('Test acurácia: ', test_acc)

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
