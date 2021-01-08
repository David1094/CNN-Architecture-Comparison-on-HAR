# -*- coding: utf-8 -*-
"""
Created on Tue Jan 5 2021 using Spyder on Windows 10

@author: David Silva
"""


"""This program trains a shallow CNN using different optimizers and a k fold 
CV frame dataset previously created. During training the program saves the 
model with the best val_accuracy and val_loss for later use. This process is 
repeated at least 5 times to avoid a lucky weight inicialition.

The path_origin variable is the one that specifies the folder where the k fold
cv frame dataset is located in the computer. The path folder must have k fold
folders with names 'Fold1', 'Fold2' ... 'FoldK', every fold folder should have
a 'Train' and 'Val' subfolders, and each of these 2 subfolders should have 51 
folders with the name of the classes of the HMDB51 dataset.
Example for Ubuntu: path = '/home/User/HMDB51_CV_frame_dataset/' 
Example for Windows: path = 'D:/User/HMDB51_CV_frame_dataset/'

The batch variable controls how many frames images the model will load at the
same time.

The epoch variable is in charge of telling the program how many times the CNN
model will train using the complete frame dataset.

The optimizadores variable is a list containing the optimizers that the CNN
will try during training. In this case I tried 5 different optimizers to see
which one gives the best acc result."""

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten
from tensorflow.keras.layers import MaxPooling2D, Conv2D, Activation
from tensorflow.keras.models import Model, Sequential
from sklearn.preprocessing import LabelBinarizer
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import DirectoryIterator
import tensorflow as tf
import numpy as np
import pickle
from os import scandir
from os.path import isdir, isfile

#Variables
batch = 16
epoch = 50
path_origin = ""
optimizadores = ["Adagrad","Adam","Nadam","RMSProp","SGD"]


def lsdir(ruta):
    return [arch.name for arch in scandir(ruta) if isdir(arch)]

def lsarch(ruta):
    return [arch.name for arch in scandir(ruta) if isfile(arch)]

def Create_label_binarizer(fold,origin_path = path_origin):
    path = origin_path + "Fold{}/".format(fold)
    print("[INFO] Creating ground truth for training...")
    print("\n")
    labels = []
    ruta = path + 'Train'
    carpetas = lsdir(ruta)
    for carpeta in carpetas:
        ruta_archivos = ruta + '/' + carpeta
        archivos = lsarch(ruta_archivos)
        for archivo in archivos:
            labels.append(carpeta)
    
    labels = np.array(labels)
    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)
    
    # serialize the label binarizer to disk
    f = open(path+"lb.pickle", "wb")
    f.write(pickle.dumps(lb))
    f.close()
    
    return path

def preprocessing(path, batch_size=batch):
    print("Normalizacion de pixeles entre 0-1")
    traingen = ImageDataGenerator(rescale=1./255)
    train = DirectoryIterator(path+'Train/',traingen,
                          target_size=(224,224),batch_size=batch, shuffle=True)

    valgen = ImageDataGenerator(rescale=1./255)
    val = DirectoryIterator(path+'Val/',valgen,
                          target_size=(224,224),batch_size=batch, shuffle=False)

    print("[INFO] Creating ground truth for training...")
    print("\n")
    labels = []
    ruta = path + 'Train'
    carpetas = lsdir(ruta)
    for carpeta in carpetas:
        ruta_archivos = ruta + '/' + carpeta
        archivos = lsarch(ruta_archivos)
        for archivo in archivos:
            labels.append(carpeta)

    labels = np.array(labels)
    # perform one-hot encoding on the labels
    lb = LabelBinarizer()
    labels = lb.fit_transform(labels)

    print("[INFO] Creating ground truth for testing...")
    print("\n")
    labelsVal = []
    ruta = path + 'Val'
    carpetas = lsdir(ruta)
    for carpeta in carpetas:
        ruta_archivos = ruta + '/' + carpeta
        archivos = lsarch(ruta_archivos)
        for archivo in archivos:
            labelsVal.append(carpeta)

    labelsVal = np.array(labelsVal)
    # perform one-hot encoding on the labels
    lb2 = LabelBinarizer()
    labelsVal = lb2.fit_transform(labelsVal)

    return train, val


for fold in range(1,4):
    path = Create_label_binarizer(fold)
    for optimizador in optimizadores:
        for corrida in range(1,6):            
            corrida = str(corrida)
            fold = str(fold)
            print("Optimizador %s Fold %s corrida %s" %(optimizador, fold, corrida))
            print("[INFO] Building model")
            
            train, val = preprocessing(path)
            
            model = Sequential()
            model.add(Input(shape=(224,224,3)))
            model.add(Conv2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            
            model.add(Conv2D(32, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            
            model.add(Conv2D(64, (3, 3)))
            model.add(Activation('relu'))
            model.add(MaxPooling2D(pool_size=(2, 2)))
            
            model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
            model.add(Dense(64, activation="relu"))
            model.add(Dropout(0.5))
            model.add(Dense(51,activation="softmax"))
            
            model = Model(inputs=model.input, outputs=model.output)
        
            print("[INFO] compiling model...")
            model.compile(loss="categorical_crossentropy", optimizer=optimizador,
                metrics=["accuracy"])


        
            callbacks = [ModelCheckpoint(filepath=path+"CNN_basica_optimizador_"+optimizador+"_fold"+fold+"_corrida_"+corrida+"_vacc.h5",
                                       monitor="val_accuracy",save_best_only=True),
                       ModelCheckpoint(filepath=path+"CNN_basica_optimizador_"+optimizador+"_fold"+fold+"_corrida_"+corrida+"_vl.h5",
                                       monitor="val_loss",save_best_only=True)]

        
            print("[INFO] training model")
            I = model.fit(train,epochs=epoch,validation_data=val)
                       

