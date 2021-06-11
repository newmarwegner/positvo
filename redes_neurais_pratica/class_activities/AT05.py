# Atividade em aula - AT05
# Autor: Newmar Wegner
# Date: 22/05/2021

import matplotlib.pyplot as plt
import numpy as np
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing import image


# Function to prepare train and validate dataset
def prepare_dataset(path_train, path_validation):
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    validation_datagen = ImageDataGenerator(rescale=1. / 255)
    
    train_generator = train_datagen.flow_from_directory(path_train,
                                                        target_size=(150, 150),
                                                        batch_size=20,
                                                        class_mode="binary"
                                                        )
    
    validation_generator = validation_datagen.flow_from_directory(
        path_validation,
        target_size=(150, 150),
        batch_size=20,
        class_mode="binary"
    )
    
    return train_datagen, validation_datagen, train_generator, validation_generator


# Function to create model
def create_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(150, 150, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model


# Function to train model
def train_model(model, train_generator, validation_generator):
    history = model.fit_generator(
        train_generator,
        steps_per_epoch=100,
        epochs=30,
        validation_data=validation_generator,
        validation_steps=50
    )
    model.save('../results/cats_and_dogs_small_1.h5')
    
    return history


# Function to evaluate training results
def evaluate_results(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs = range(1, len(acc) + 1)
    
    return acc, val_acc, loss, val_loss, epochs


# Function to plot results
def plot_results(epochs, acc, val_acc, loss, val_loss):
    plt.plot(epochs, acc, 'bo', label='Training acc')
    plt.plot(epochs, val_acc, 'b', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.figure()
    
    plt.plot(epochs, loss, 'bo', label='Training loss')
    plt.plot(epochs, val_loss, 'b', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()
    plt.show()
    
    return

# Function to load model
def execute_model_save(path):
    model = load_model(path)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    
    return model

# Function to predict images
def classify_images(img_path_classify):
    img = image.load_img(img_path_classify, target_size=(img_width, img_height))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    images = np.vstack([x])
    
    return model_saved.predict(images)

if __name__ == '__main__':
    ##### Create model and get results ###########
    path_train = './data/cats_dogs/catdog_train'
    path_validation = './data/cats_dogs/catdog_validation'
    train_datagen, validation_datagen, train_generator, validation_generator = prepare_dataset(path_train,
                                                                                               path_validation)
    model = create_model()
    model.summary()
    history = train_model(model, train_generator, validation_generator)
    acc, val_acc, loss, val_loss, epochs = evaluate_results(history)
    plot_results(epochs, acc, val_acc, loss, val_loss)
    
    ###### Classify images with model saved ############
    # dimensions of our images
    img_width, img_height = 150, 150
    
    # load the model we saved
    model_saved = execute_model_save('../results/cats_and_dogs_small_1.h5')
    
    # predicting images
    img_path_classify = './data/cats_dogs/catdog_test/dogs/dog.1422.jpg'
    print(classify_images(img_path_classify))
