# -*- coding: utf-8 -*-
"""
Created on Tue Jan 5 2021 using Spyder on Windows 10

@author: David Silva
"""


"""This program loads a prebuild CNN architecture from keras library, then the 
program trains the CNN using a frame dataset previously created and during 
training the program saves the model with the best val_accuracy for later use.
This process is repeated at least 3 times to avoid a lucky weight inicialition.

The cv variable indicates whether to use the training and validation folders
of a fold (cv=1) or to use the training and test folders of a split (cv=0).

The path variable is the one that specifies the folder where the frame dataset
is located in the computer. The path folder must have two main folders: 'Train'
and 'Test'. Any of these 2 folders must contain the 51 folders related to the 
classes of the HMDB51 dataset. Each of the 51 folders must contain the frames 
related to the videos of that class which were previously generated.
Example for Ubuntu: path = '/home/User/HMDB51_frame_dataset/' 
Example for Windows: path = 'D:/User/HMDB51_frame_dataset/'

The batch variable controls how many frames images the model will load at the
same time.

The epoch variable is in charge of telling the program how many times the CNN
model will train using the complete frame dataset.

The imagenet variable can have integer values of 0 and 1 and is used when the
program load the CNN architecture, it specifies whether or not to use the 
pretrained imagenet weights.

The network variable specifies the name of the architecture that the user wants
to load. It can be any of these options [Xception, InceptionV3, ResNet152,
MobileNetV2, NASNetMobile, DenseNet201, EfficientNetB0, EfficientNetB3]"""


from efficientnet.tfkeras import EfficientNetB0, EfficientNetB3
from tensorflow.keras.applications import ResNet152
from tensorflow.keras.applications import InceptionV3, Xception
from tensorflow.keras.applications import MobileNetV2, NASNetMobile
from tensorflow.keras.applications import DenseNet201
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_caffe
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_tf
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import DirectoryIterator
from sklearn.preprocessing import LabelBinarizer
import tensorflow as tf
import numpy as np
import pickle
from os import scandir
from os.path import isdir, isfile
import time

#Variables
cv = 0
path = ""
batch = 16
epoch = 10
imagenet = 1
network = "Xception"


def lsdir(ruta):
    return [arch.name for arch in scandir(ruta) if isdir(arch)]

def lsarch(ruta):
    return [arch.name for arch in scandir(ruta) if isfile(arch)]

def preprocessing(network,path=path, batch_size=batch, mode=cv):
    tf_preprocessing = ["Xception", "InceptionV3", "NASNetMobile", "MobileNetV2", "ResNet152V2"]
    network299 = ["Xception", "InceptionV3"]
    if network in tf_preprocessing:
        preprocess = preprocess_input_tf
        modo = "tf"
    else:
        preprocess = preprocess_input_caffe
        modo = "caffe"
    print("Preprocesamiento tipo %s" %(modo))
    if network in network299:
        resize = (299,299)
    else:
        resize = (224,224)
    if mode == 0:
        folder_name = "Test"
    else:
        folder_name = "Val"
    traingen = ImageDataGenerator(preprocessing_function=preprocess)
    train = DirectoryIterator(path+'Train/',traingen,
                          target_size=resize,batch_size=batch, shuffle=True)

    testgen = ImageDataGenerator(preprocessing_function=preprocess)
    test = DirectoryIterator(path+folder_name+'/',testgen,
                          target_size=resize,batch_size=batch, shuffle=False)

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
    labelsTest = []
    ruta = path + folder_name
    carpetas = lsdir(ruta)
    for carpeta in carpetas:
        ruta_archivos = ruta + '/' + carpeta
        archivos = lsarch(ruta_archivos)
        for archivo in archivos:
            labelsTest.append(carpeta)

    labelsTest = np.array(labelsTest)
    # perform one-hot encoding on the labels
    lb2 = LabelBinarizer()
    labelsTest = lb2.fit_transform(labelsTest)

    return train, test, lb2, labelsTest

def preprocessing_EfficcientNet(path=path, batch_size=batch, mode=cv):
    print("Normalizacion de pixeles entre 0-1")
    if mode == 0:
        folder_name = "Test"
    else:
        folder_name = "Val"
    traingen = ImageDataGenerator(rescale=1./255)
    train = DirectoryIterator(path+'Train/',traingen,
                          target_size=(224,224),batch_size=batch, shuffle=True)

    testgen = ImageDataGenerator(rescale=1./255)
    test = DirectoryIterator(path+folder_name+'/',testgen,
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
    labelsTest = []
    ruta = path + folder_name
    carpetas = lsdir(ruta)
    for carpeta in carpetas:
        ruta_archivos = ruta + '/' + carpeta
        archivos = lsarch(ruta_archivos)
        for archivo in archivos:
            labelsTest.append(carpeta)

    labelsTest = np.array(labelsTest)
    # perform one-hot encoding on the labels
    lb2 = LabelBinarizer()
    labelsTest = lb2.fit_transform(labelsTest)

    return train, test, lb2, labelsTest


if imagenet == 0:
    pretraining = "None"
else:
    pretraining = "imagenet"

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

for corrida in range(1,4):
    if network == "Xception":
        print(network)
        train, test, lb2, labelsTest = preprocessing(network)
        baseModel = Xception(weights=pretraining, include_top=False,input_tensor=Input(shape=(299, 299, 3)),pooling="avg")
        headModel = baseModel.output
        headModel = Dense(51, activation='softmax')(headModel)
    elif network == "ResNet152":
        print(network)
        train, test, lb2, labelsTest = preprocessing(network)
        baseModel = ResNet152(weights=pretraining, include_top=False,input_tensor=Input(shape=(224, 224, 3)),pooling="avg")
        headModel = baseModel.output
        headModel = Dense(51, activation='softmax')(headModel)
    elif network == "InceptionV3":
        print(network)
        train, test, lb2, labelsTest = preprocessing(network)
        baseModel = InceptionV3(weights=pretraining, include_top=False,input_tensor=Input(shape=(299, 299, 3)),pooling="avg")
        headModel = baseModel.output
        headModel = Dense(51, activation='softmax')(headModel)
    elif network == "MobileNetV2":
        print(network)
        train, test, lb2, labelsTest = preprocessing(network)
        baseModel = MobileNetV2(weights=pretraining, include_top=False,input_tensor=Input(shape=(224, 224, 3)),pooling="avg")
        headModel = baseModel.output
        headModel = Dense(51, activation='softmax',use_bias=True)(headModel)
    elif network == "DenseNet201":
        print(network)
        train, test, lb2, labelsTest = preprocessing_EfficcientNet()
        baseModel = DenseNet201(weights=pretraining, include_top=False,input_tensor=Input(shape=(224, 224, 3)),pooling="avg")
        headModel = baseModel.output
        headModel = Dense(51, activation='softmax')(headModel)
    elif network == "NASNetMobile":
        print(network)
        train, test, lb2, labelsTest = preprocessing(network)
        baseModel = NASNetMobile(weights=pretraining, include_top=False,input_tensor=Input(shape=(224, 224, 3)),pooling="avg")
        headModel = baseModel.output
        headModel = Dense(51, activation='softmax')(headModel)
    elif network == "EfficientNetB0":
        print(network)
        train, test, lb2, labelsTest = preprocessing_EfficcientNet()
        baseModel = EfficientNetB0(weights=pretraining,include_top=False,input_tensor=Input(shape=(224, 224, 3)),pooling="avg")
        headModel = baseModel.output
        headModel = Dropout(0.2)(headModel)
        headModel = Dense(51, activation='softmax')(headModel)
    else:
        print(network)
        train, test, lb2, labelsTest = preprocessing_EfficcientNet()
        baseModel = EfficientNetB3(weights=pretraining,include_top=False,input_tensor=Input(shape=(224, 224, 3)),pooling="avg")
        headModel = baseModel.output
        headModel = Dropout(0.2)(headModel)
        headModel = Dense(51, activation='softmax')(headModel)


    corrida = str(corrida)
    print("corrida %s" %(corrida))
    model = Model(inputs=baseModel.input, outputs=headModel)

    for layer in model.layers:
        layer.trainable = True

    print("[INFO] compiling model...")
    opt = SGD()
    model.compile(loss="categorical_crossentropy", optimizer=opt,
        metrics=["accuracy",tf.keras.metrics.TopKCategoricalAccuracy(name="top_5_acc")])

    callbacks = [ModelCheckpoint(filepath=path+network+"_corrida_"+corrida+"_vacc.h5",monitor="val_accuracy",save_best_only=True)]

    print("[INFO] training model")
    time1 = time.time()
    I = model.fit(train,epochs=epoch,validation_data=test,callbacks=callbacks)
    
    time2 = time.time()
    time_total = time2 - time1
    print("El tiempo total del entrenamiento en segundos de la red %s es de: %d" %(network,time_total))
