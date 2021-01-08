# -*- coding: utf-8 -*-
"""
Created on Tue Jan 5 2021 using Spyder on Windows 10

@author: David Silva
"""


"""This program makes an ensemble loading 3 or 5 pretrained CNN arhitecture
model, and test the ensemble using different concensus function and the frames 
of the test videos of the set of the HMDB51 video dataset that was used during
training.

The concensus_function variable is responsable for telling the program how
the ensamble will predict the label of a video. It can be either using simple
voting (concensus_funcion=0), using weighted votes (concensus_funcion=1) or 
using the average of the predictions (concensus_function=2).

The number_of_models variables tell the program whether to use 3 or 5 models
to make an ensemble.

The all_frames variable is in charge of telling the program whether to use only
the extracted frames of the test videos that were created before in the process
of making the frame dataset (all_frames=0) or to use all the frames of the test
videos (all_frames=1) to make the final prediction of the video label.

The n_frames_p_video variable is only used when all_frames=0 and it tells the
program to use a specific number of frames from the frame dataset. That way the
program only average the predictions of the frames related to the same video.

The red variable specifies the name of the architecture that was use during
training and is useful to specify the program what type of the preprocessing 
the test frames need. It can be any of these options [Xception, InceptionV3,
ResNet152, MobileNetV2, NASNetMobile, DenseNet201, EfficientNetB0, 
EfficientNetB3]

The lbpickle variable is the path of the label binarizer that was created
during training.
Example for Ubuntu: path = '/home/User/HMDB51_save_models/lb.pickle' 
Example for Windows: path = 'D:/User/HMDB51_save_models/lb.pickle'

The best_5_models variable refers to a list containing the path of the 5 best
CNN models to use in an ensemble and the first element of the list must be the
best model, the second one the second best model and so on. Every path element 
must be writed like the next examples.
Example for Ubuntu: path = '/home/User/HMDB51_save_models/model.h5' 
Example for Windows: path = 'D:/User/HMDB51_save_models/model.h5'

The best_5_architectures variable refers to a list containing the name of the 
5 best CNN architectures and the first element of the list must be the best 
architecture, the second one the second best architecture and so on. Must
ensure that the CNN names matches with the CNN that was used in every model
of the best_5_models variable. Name can be any of these options [Xception, 
InceptionV3, ResNet152, MobileNetV2, NASNetMobile, DenseNet201, EfficientNetB0,
EfficientNetB3]

The path_frame_dataset variable is the one that specifies the folder where the 
test frames of the frame dataset of a certain set are located in the computer. 
The folder must contain the 51 folders related to the classes of the HMDB51 
dataset. Each of the 51 folders must contain the frames related to the videos 
of that class which were previously generated.
Example for Ubuntu: path = '/home/User/HMDB51_set1_frame_dataset/Test/' 
Example for Windows: path = 'D:/User/HMDB51_set1_frame_dataset/Test/'

The path_video_dataset variable is the one that specifies the folder where the 
test videos of a certain set are located in the computer. The folder must 
contain the 51 folders related to the classes of the HMDB51 dataset.
Example for Ubuntu: path = '/home/User/HMDB51_set1_videos/Test/' 
Example for Windows: path = 'D:/User/HMDB51_set1_videos/Test/'
"""

from efficientnet.tfkeras import EfficientNetB0
from os import scandir
from os.path import abspath
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.vgg16 import preprocess_input as preprocess_input_caffe
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_tf
from collections import deque
import numpy as np
import pickle
import cv2

#Variables
concensus_function = 0
number_of_models = 3
all_frames = 1
n_frames_p_video = 20
red = ""
lbpickle = ""
best_5_models = ["","","","",""]
best_5_architectures = ["","","","",""]
path_frame_dataset = ""
path_video_dataset = ""

#Constants
labels = ["brush_hair","cartwheel","catch","chew","clap","climb","climb_stairs",
    "dive","draw_sword","dribble","drink","eat","fall_floor","fencing","flic_flac",
    "golf","handstand","hit","hug","jump","kick","kick_ball","kiss","laugh","pick",
    "pour","pullup","punch","push","pushup","ride_bike","ride_horse","run","shake_hands",
    "shoot_ball","shoot_bow","shoot_gun","sit","situp","smile","smoke","somersault",
    "stand","swing_baseball","sword","sword_exercise","talk","throw","turn","walk","wave"]


def ls(ruta):
    return [abspath(arch.path) for arch in scandir(ruta) if arch.is_file()]

def lsarch(ruta):
    return [arch.name for arch in scandir(ruta) if arch.is_file()]

def preprocessing(network,frame):
    network_preprocessing = ["Xception", "InceptionV3", "NASNetMobile", "MobileNetV2", "ResNet152", "VGG19"]
    if network in network_preprocessing:
        frame = preprocessing1(network,frame)
    else:
        frame = preprocessing2(network,frame)
    return frame

def preprocessing1(network,frame):
    tf_preprocessing = ["Xception", "InceptionV3", "NASNetMobile", "MobileNetV2"]
    network299 = ["Xception", "InceptionV3"]
    if network in tf_preprocessing:
        preprocess = preprocess_input_tf
    else:
        preprocess = preprocess_input_caffe
    if network in network299:
        resize = (299,299)
    else:
        resize = (224,224)
    frame = cv2.resize(frame,resize)
    frame = preprocess(frame)

    return frame

def preprocessing2(network, frame):
    resize = (224,224)
    frame = cv2.resize(frame,resize)
    frame = frame.astype("float32")
    frame /= 255
    return frame


lb = pickle.loads(open(lbpickle, "rb").read())
if all_frames == 1:
    if concensus_function == 0: #simple voting
        if number_of_models == 3:
            model1 = load_model(best_5_models[0])
            model2 = load_model(best_5_models[1])
            model3 = load_model(best_5_models[2])
            total = 0
            for clase in labels:
                etiqueta = path_video_dataset + str(clase)
                videos = ls(ruta=etiqueta)
                clases = {"brush_hair":0,"cartwheel":0,"catch":0,"chew":0,"clap":0,"climb":0,"climb_stairs":0,
                "dive":0,"draw_sword":0,"dribble":0,"drink":0,"eat":0,"fall_floor":0,"fencing":0,"flic_flac":0,
                "golf":0,"handstand":0,"hit":0,"hug":0,"jump":0,"kick":0,"kick_ball":0,"kiss":0,"laugh":0,"pick":0,
                "pour":0,"pullup":0,"punch":0,"push":0,"pushup":0,"ride_bike":0,"ride_horse":0,"run":0,"shake_hands":0,
                "shoot_ball":0,"shoot_bow":0,"shoot_gun":0,"sit":0,"situp":0,"smile":0,"smoke":0,"somersault":0,
                "stand":0,"swing_baseball":0,"sword":0,"sword_exercise":0,"talk":0,"throw":0,"turn":0,"walk":0,"wave":0}
            
                for clip in range(0,len(videos)):
                    vs = cv2.VideoCapture(videos[clip])
            
                    counts = 0
                    while(True):
                        (grabbed, frame) = vs.read()
            
                        if not grabbed:
                            break
            
                        counts += 1
                    Q = deque(maxlen=counts)
                    R = deque(maxlen=counts)
                    S = deque(maxlen=counts)
            
                    vs2 = cv2.VideoCapture(videos[clip])
            
                    while(True):
                        (grabbed, frame) = vs2.read()
            
                        if not grabbed:
                            break
            
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame1 = preprocessing(best_5_architectures[0],frame)
                        frame2 = preprocessing(best_5_architectures[1],frame)
                        frame3 = preprocessing(best_5_architectures[2],frame)
            
            
                        preds1 = model1.predict(np.expand_dims(frame1, axis=0))[0]
                        preds2 = model2.predict(np.expand_dims(frame2, axis=0))[0]
                        preds3 = model3.predict(np.expand_dims(frame3, axis=0))[0]
                        Q.append(preds1)
                        R.append(preds2)
                        S.append(preds3)
            
                    results1 = np.array(Q).mean(axis=0)
                    results2 = np.array(R).mean(axis=0)
                    results3 = np.array(S).mean(axis=0)
                    i1 = np.argmax(results1)
                    i2 = np.argmax(results2)
                    i3 = np.argmax(results3)
                    label1 = lb.classes_[i1]
                    label2 = lb.classes_[i2]
                    label3 = lb.classes_[i3]
                    labels_models = [label1,label2,label3]
                    predictions = dict()
                    for label in labels_models:
                        if label not in predictions.keys():
                            predictions.update({label:1})
                        else:
                            predictions.update({label:predictions.get(label)+1})
            
                    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
                    true_label = sorted_predictions[0][0]
                    clases[true_label] += 1
                    vs.release()
                    vs2.release()
            
                total += clases[clase]
                print("El numero de videos de la clase %s es de %d" %(str(clase),len(videos)))    
                print("El numero de videos que fue predecido como clase %s es de %d" %(str(clase),clases[clase]))
                print("El numero de videos que fue predecido incorrectamente es de %d" %(len(videos)-clases[clase]))
                print("\n")
            print("Numero total de aciertos: %d" %(total))
            print("Precision: %f" %((total/1530)*100)) 
        else:
            model1 = load_model(best_5_models[0])
            model2 = load_model(best_5_models[1])
            model3 = load_model(best_5_models[2])
            model4 = load_model(best_5_models[3])
            model5 = load_model(best_5_models[4])
            total = 0
            for clase in labels:
                etiqueta = path_video_dataset + str(clase)
                videos = ls(ruta=etiqueta)
                clases = {"brush_hair":0,"cartwheel":0,"catch":0,"chew":0,"clap":0,"climb":0,"climb_stairs":0,
                "dive":0,"draw_sword":0,"dribble":0,"drink":0,"eat":0,"fall_floor":0,"fencing":0,"flic_flac":0,
                "golf":0,"handstand":0,"hit":0,"hug":0,"jump":0,"kick":0,"kick_ball":0,"kiss":0,"laugh":0,"pick":0,
                "pour":0,"pullup":0,"punch":0,"push":0,"pushup":0,"ride_bike":0,"ride_horse":0,"run":0,"shake_hands":0,
                "shoot_ball":0,"shoot_bow":0,"shoot_gun":0,"sit":0,"situp":0,"smile":0,"smoke":0,"somersault":0,
                "stand":0,"swing_baseball":0,"sword":0,"sword_exercise":0,"talk":0,"throw":0,"turn":0,"walk":0,"wave":0}
            
                for clip in range(0,len(videos)):
                    vs = cv2.VideoCapture(videos[clip])
            
                    counts = 0
                    while(True):
                        (grabbed, frame) = vs.read()
            
                        if not grabbed:
                            break
            
                        counts += 1
                    Q = deque(maxlen=counts)
                    R = deque(maxlen=counts)
                    S = deque(maxlen=counts)
                    T = deque(maxlen=counts)
                    U = deque(maxlen=counts)
            
                    vs2 = cv2.VideoCapture(videos[clip])
            
                    while(True):
                        (grabbed, frame) = vs2.read()
            
                        if not grabbed:
                            break
            
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame1 = preprocessing(best_5_architectures[0],frame)
                        frame2 = preprocessing(best_5_architectures[1],frame)
                        frame3 = preprocessing(best_5_architectures[2],frame)
                        frame4 = preprocessing(best_5_architectures[3],frame)
                        frame5 = preprocessing(best_5_architectures[4],frame)
            
            
                        preds1 = model1.predict(np.expand_dims(frame1, axis=0))[0]
                        preds2 = model2.predict(np.expand_dims(frame2, axis=0))[0]
                        preds3 = model3.predict(np.expand_dims(frame3, axis=0))[0]
                        preds4 = model4.predict(np.expand_dims(frame4, axis=0))[0]
                        preds5 = model5.predict(np.expand_dims(frame5, axis=0))[0]
                        Q.append(preds1)
                        R.append(preds2)
                        S.append(preds3)
                        T.append(preds4)
                        U.append(preds5)
            
                    results1 = np.array(Q).mean(axis=0)
                    results2 = np.array(R).mean(axis=0)
                    results3 = np.array(S).mean(axis=0)
                    results4 = np.array(T).mean(axis=0)
                    results5 = np.array(U).mean(axis=0)
                    i1 = np.argmax(results1)
                    i2 = np.argmax(results2)
                    i3 = np.argmax(results3)
                    i4 = np.argmax(results4)
                    i5 = np.argmax(results5)
                    label1 = lb.classes_[i1]
                    label2 = lb.classes_[i2]
                    label3 = lb.classes_[i3]
                    label4 = lb.classes_[i4]
                    label5 = lb.classes_[i5]
                    labels_models = [label1,label2,label3,label4,label5]
                    predictions = dict()
                    for label in labels_models:
                        if label not in predictions.keys():
                            predictions.update({label:1})
                        else:
                            predictions.update({label:predictions.get(label)+1})
            
                    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
                    true_label = sorted_predictions[0][0]
                    clases[true_label] += 1
                    vs.release()
                    vs2.release()
            
                total += clases[clase]
                print("El numero de videos de la clase %s es de %d" %(str(clase),len(videos)))    
                print("El numero de videos que fue predecido como clase %s es de %d" %(str(clase),clases[clase]))
                print("El numero de videos que fue predecido incorrectamente es de %d" %(len(videos)-clases[clase]))
                print("\n")
            print("Numero total de aciertos: %d" %(total))
            print("Precision: %f" %((total/1530)*100))           
    elif concensus_function == 1: #weighted voting
        if number_of_models == 3:
            model1 = load_model(best_5_models[0])
            model2 = load_model(best_5_models[1])
            model3 = load_model(best_5_models[2])
            total = 0
            for clase in labels:
                etiqueta = path_video_dataset + str(clase)
                videos = ls(ruta=etiqueta)
                clases = {"brush_hair":0,"cartwheel":0,"catch":0,"chew":0,"clap":0,"climb":0,"climb_stairs":0,
                "dive":0,"draw_sword":0,"dribble":0,"drink":0,"eat":0,"fall_floor":0,"fencing":0,"flic_flac":0,
                "golf":0,"handstand":0,"hit":0,"hug":0,"jump":0,"kick":0,"kick_ball":0,"kiss":0,"laugh":0,"pick":0,
                "pour":0,"pullup":0,"punch":0,"push":0,"pushup":0,"ride_bike":0,"ride_horse":0,"run":0,"shake_hands":0,
                "shoot_ball":0,"shoot_bow":0,"shoot_gun":0,"sit":0,"situp":0,"smile":0,"smoke":0,"somersault":0,
                "stand":0,"swing_baseball":0,"sword":0,"sword_exercise":0,"talk":0,"throw":0,"turn":0,"walk":0,"wave":0}
            
                for clip in range(0,len(videos)):
                    vs = cv2.VideoCapture(videos[clip])
            
                    counts = 0
                    while(True):
                        (grabbed, frame) = vs.read()
            
                        if not grabbed:
                            break
            
                        counts += 1
                    Q = deque(maxlen=counts)
                    R = deque(maxlen=counts)
                    S = deque(maxlen=counts)
            
                    vs2 = cv2.VideoCapture(videos[clip])
            
                    while(True):
                        (grabbed, frame) = vs2.read()
            
                        if not grabbed:
                            break
            
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame1 = preprocessing(best_5_architectures[0],frame)
                        frame2 = preprocessing(best_5_architectures[1],frame)
                        frame3 = preprocessing(best_5_architectures[2],frame)
            
            
                        preds1 = model1.predict(np.expand_dims(frame1, axis=0))[0]
                        preds2 = model2.predict(np.expand_dims(frame2, axis=0))[0]
                        preds3 = model3.predict(np.expand_dims(frame3, axis=0))[0]
                        Q.append(preds1)
                        R.append(preds2)
                        S.append(preds3)
            
                    results1 = np.array(Q).mean(axis=0)
                    results2 = np.array(R).mean(axis=0)
                    results3 = np.array(S).mean(axis=0)
                    i1 = np.argmax(results1)
                    i2 = np.argmax(results2)
                    i3 = np.argmax(results3)
                    label1 = lb.classes_[i1]
                    label2 = lb.classes_[i2]
                    label3 = lb.classes_[i3]
                    labels_models = [label1,label2,label3]
                    predictions = dict()
                    for i,label in enumerate(labels_models):
                        if i == 0:
                            value = 3
                        elif i == 1:
                            value = 2
                        else:
                            value = 1
                        if label not in predictions.keys():
                            predictions.update({label:value})
                        else:
                            predictions.update({label:predictions.get(label)+value})
        
                    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
                    true_label = sorted_predictions[0][0]
                    clases[true_label] += 1
                    vs.release()
                    vs2.release()
            
                total += clases[clase]
                print("El numero de videos de la clase %s es de %d" %(str(clase),len(videos)))    
                print("El numero de videos que fue predecido como clase %s es de %d" %(str(clase),clases[clase]))
                print("El numero de videos que fue predecido incorrectamente es de %d" %(len(videos)-clases[clase]))
                print("\n")
            print("Numero total de aciertos: %d" %(total))
            print("Precision: %f" %((total/1530)*100)) 
        else:
            model1 = load_model(best_5_models[0])
            model2 = load_model(best_5_models[1])
            model3 = load_model(best_5_models[2])
            model4 = load_model(best_5_models[3])
            model5 = load_model(best_5_models[4])
            total = 0
            for clase in labels:
                etiqueta = path_video_dataset + str(clase)
                videos = ls(ruta=etiqueta)
                clases = {"brush_hair":0,"cartwheel":0,"catch":0,"chew":0,"clap":0,"climb":0,"climb_stairs":0,
                "dive":0,"draw_sword":0,"dribble":0,"drink":0,"eat":0,"fall_floor":0,"fencing":0,"flic_flac":0,
                "golf":0,"handstand":0,"hit":0,"hug":0,"jump":0,"kick":0,"kick_ball":0,"kiss":0,"laugh":0,"pick":0,
                "pour":0,"pullup":0,"punch":0,"push":0,"pushup":0,"ride_bike":0,"ride_horse":0,"run":0,"shake_hands":0,
                "shoot_ball":0,"shoot_bow":0,"shoot_gun":0,"sit":0,"situp":0,"smile":0,"smoke":0,"somersault":0,
                "stand":0,"swing_baseball":0,"sword":0,"sword_exercise":0,"talk":0,"throw":0,"turn":0,"walk":0,"wave":0}
            
                for clip in range(0,len(videos)):
                    vs = cv2.VideoCapture(videos[clip])
            
                    counts = 0
                    while(True):
                        (grabbed, frame) = vs.read()
            
                        if not grabbed:
                            break
            
                        counts += 1
                    Q = deque(maxlen=counts)
                    R = deque(maxlen=counts)
                    S = deque(maxlen=counts)
                    T = deque(maxlen=counts)
                    U = deque(maxlen=counts)
            
                    vs2 = cv2.VideoCapture(videos[clip])
            
                    while(True):
                        (grabbed, frame) = vs2.read()
            
                        if not grabbed:
                            break
            
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame1 = preprocessing(best_5_architectures[0],frame)
                        frame2 = preprocessing(best_5_architectures[1],frame)
                        frame3 = preprocessing(best_5_architectures[2],frame)
                        frame4 = preprocessing(best_5_architectures[3],frame)
                        frame5 = preprocessing(best_5_architectures[4],frame)
            
            
                        preds1 = model1.predict(np.expand_dims(frame1, axis=0))[0]
                        preds2 = model2.predict(np.expand_dims(frame2, axis=0))[0]
                        preds3 = model3.predict(np.expand_dims(frame3, axis=0))[0]
                        preds4 = model4.predict(np.expand_dims(frame4, axis=0))[0]
                        preds5 = model5.predict(np.expand_dims(frame5, axis=0))[0]
                        Q.append(preds1)
                        R.append(preds2)
                        S.append(preds3)
                        T.append(preds4)
                        U.append(preds5)
            
                    results1 = np.array(Q).mean(axis=0)
                    results2 = np.array(R).mean(axis=0)
                    results3 = np.array(S).mean(axis=0)
                    results4 = np.array(T).mean(axis=0)
                    results5 = np.array(U).mean(axis=0)
                    i1 = np.argmax(results1)
                    i2 = np.argmax(results2)
                    i3 = np.argmax(results3)
                    i4 = np.argmax(results4)
                    i5 = np.argmax(results5)
                    label1 = lb.classes_[i1]
                    label2 = lb.classes_[i2]
                    label3 = lb.classes_[i3]
                    label4 = lb.classes_[i4]
                    label5 = lb.classes_[i5]
                    labels_models = [label1,label2,label3,label4,label5]
                    predictions = dict()
                    for i,label in enumerate(labels_models):
                        if i == 0:
                            value = 5
                        elif i == 1:
                            value = 4
                        elif i == 2:
                            value = 3
                        elif i == 3:
                            value = 2
                        else:
                            value = 1
                        if label not in predictions.keys():
                            predictions.update({label:value})
                        else:
                            predictions.update({label:predictions.get(label)+value})
        
                    sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
        
                    true_label = sorted_predictions[0][0]
                    clases[true_label] += 1
                    vs.release()
                    vs2.release()
            
                total += clases[clase]
                print("El numero de videos de la clase %s es de %d" %(str(clase),len(videos)))    
                print("El numero de videos que fue predecido como clase %s es de %d" %(str(clase),clases[clase]))
                print("El numero de videos que fue predecido incorrectamente es de %d" %(len(videos)-clases[clase]))
                print("\n")
            print("Numero total de aciertos: %d" %(total))
            print("Precision: %f" %((total/1530)*100)) 
    else:
        if number_of_models == 3: #average of predictions
            model1 = load_model(best_5_models[0])
            model2 = load_model(best_5_models[1])
            model3 = load_model(best_5_models[2])
            total = 0
            for clase in labels:
                etiqueta = path_video_dataset + str(clase)
                videos = ls(ruta=etiqueta)
                clases = {"brush_hair":0,"cartwheel":0,"catch":0,"chew":0,"clap":0,"climb":0,"climb_stairs":0,
                "dive":0,"draw_sword":0,"dribble":0,"drink":0,"eat":0,"fall_floor":0,"fencing":0,"flic_flac":0,
                "golf":0,"handstand":0,"hit":0,"hug":0,"jump":0,"kick":0,"kick_ball":0,"kiss":0,"laugh":0,"pick":0,
                "pour":0,"pullup":0,"punch":0,"push":0,"pushup":0,"ride_bike":0,"ride_horse":0,"run":0,"shake_hands":0,
                "shoot_ball":0,"shoot_bow":0,"shoot_gun":0,"sit":0,"situp":0,"smile":0,"smoke":0,"somersault":0,
                "stand":0,"swing_baseball":0,"sword":0,"sword_exercise":0,"talk":0,"throw":0,"turn":0,"walk":0,"wave":0}
            
                for clip in range(0,len(videos)):
                    vs = cv2.VideoCapture(videos[clip])
            
                    counts = 0
                    while(True):
                        (grabbed, frame) = vs.read()
            
                        if not grabbed:
                            break
            
                        counts += 1
                    Q = deque(maxlen=counts*3)
            
                    vs2 = cv2.VideoCapture(videos[clip])
            
                    while(True):
                        (grabbed, frame) = vs2.read()
            
                        if not grabbed:
                            break
            
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame1 = preprocessing(best_5_architectures[0],frame)
                        frame2 = preprocessing(best_5_architectures[1],frame)
                        frame3 = preprocessing(best_5_architectures[2],frame)
            
            
                        preds1 = model1.predict(np.expand_dims(frame1, axis=0))[0]
                        preds2 = model2.predict(np.expand_dims(frame2, axis=0))[0]
                        preds3 = model3.predict(np.expand_dims(frame3, axis=0))[0]
                        Q.append(preds1)
                        Q.append(preds2)
                        Q.append(preds3)
            
                    results = np.array(Q).mean(axis=0)
                    i = np.argmax(results)
                    label = lb.classes_[i]
                    clases[label] += 1
                    vs.release()
                    vs2.release()
            
                total += clases[clase]
                print("El numero de videos de la clase %s es de %d" %(str(clase),len(videos)))    
                print("El numero de videos que fue predecido como clase %s es de %d" %(str(clase),clases[clase]))
                print("El numero de videos que fue predecido incorrectamente es de %d" %(len(videos)-clases[clase]))
                print("\n")
            print("Numero total de aciertos: %d" %(total))
            print("Precision: %f" %((total/1530)*100))            
        else:
            model1 = load_model(best_5_models[0])
            model2 = load_model(best_5_models[1])
            model3 = load_model(best_5_models[2])
            model4 = load_model(best_5_models[3])
            model5 = load_model(best_5_models[4])
            total = 0
            for clase in labels:
                etiqueta = path_video_dataset + str(clase)
                videos = ls(ruta=etiqueta)
                clases = {"brush_hair":0,"cartwheel":0,"catch":0,"chew":0,"clap":0,"climb":0,"climb_stairs":0,
                "dive":0,"draw_sword":0,"dribble":0,"drink":0,"eat":0,"fall_floor":0,"fencing":0,"flic_flac":0,
                "golf":0,"handstand":0,"hit":0,"hug":0,"jump":0,"kick":0,"kick_ball":0,"kiss":0,"laugh":0,"pick":0,
                "pour":0,"pullup":0,"punch":0,"push":0,"pushup":0,"ride_bike":0,"ride_horse":0,"run":0,"shake_hands":0,
                "shoot_ball":0,"shoot_bow":0,"shoot_gun":0,"sit":0,"situp":0,"smile":0,"smoke":0,"somersault":0,
                "stand":0,"swing_baseball":0,"sword":0,"sword_exercise":0,"talk":0,"throw":0,"turn":0,"walk":0,"wave":0}
            
                for clip in range(0,len(videos)):
                    vs = cv2.VideoCapture(videos[clip])
            
                    counts = 0
                    while(True):
                        (grabbed, frame) = vs.read()
            
                        if not grabbed:
                            break
            
                        counts += 1
                    Q = deque(maxlen=counts*5)
            
                    vs2 = cv2.VideoCapture(videos[clip])
            
                    while(True):
                        (grabbed, frame) = vs2.read()
            
                        if not grabbed:
                            break
            
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame1 = preprocessing(best_5_architectures[0],frame)
                        frame2 = preprocessing(best_5_architectures[1],frame)
                        frame3 = preprocessing(best_5_architectures[2],frame)
                        frame4 = preprocessing(best_5_architectures[3],frame)
                        frame5 = preprocessing(best_5_architectures[4],frame)
            
            
                        preds1 = model1.predict(np.expand_dims(frame1, axis=0))[0]
                        preds2 = model2.predict(np.expand_dims(frame2, axis=0))[0]
                        preds3 = model3.predict(np.expand_dims(frame3, axis=0))[0]
                        preds4 = model4.predict(np.expand_dims(frame4, axis=0))[0]
                        preds5 = model5.predict(np.expand_dims(frame5, axis=0))[0]
                        Q.append(preds1)
                        Q.append(preds2)
                        Q.append(preds3)
                        Q.append(preds4)
                        Q.append(preds5)
            
                    results = np.array(Q).mean(axis=0)
                    i = np.argmax(results)
                    label = lb.classes_[i]
                    clases[label] += 1
                    vs.release()
                    vs2.release()
            
                total += clases[clase]
                print("El numero de videos de la clase %s es de %d" %(str(clase),len(videos)))    
                print("El numero de videos que fue predecido como clase %s es de %d" %(str(clase),clases[clase]))
                print("El numero de videos que fue predecido incorrectamente es de %d" %(len(videos)-clases[clase]))
                print("\n")
            print("Numero total de aciertos: %d" %(total))
            print("Precision: %f" %((total/1530)*100))
else:
    if concensus_function == 0: #simple voting
        if number_of_models == 3:
            model1 = load_model(best_5_models[0])
            model2 = load_model(best_5_models[1])
            model3 = load_model(best_5_models[2])
            total = 0
            for clase in labels:
                etiqueta = path_frame_dataset + str(clase)
                ruta_frames = ls(ruta=etiqueta)
                clases = {"brush_hair":0,"cartwheel":0,"catch":0,"chew":0,"clap":0,"climb":0,"climb_stairs":0,
                "dive":0,"draw_sword":0,"dribble":0,"drink":0,"eat":0,"fall_floor":0,"fencing":0,"flic_flac":0,
                "golf":0,"handstand":0,"hit":0,"hug":0,"jump":0,"kick":0,"kick_ball":0,"kiss":0,"laugh":0,"pick":0,
                "pour":0,"pullup":0,"punch":0,"push":0,"pushup":0,"ride_bike":0,"ride_horse":0,"run":0,"shake_hands":0,
                "shoot_ball":0,"shoot_bow":0,"shoot_gun":0,"sit":0,"situp":0,"smile":0,"smoke":0,"somersault":0,
                "stand":0,"swing_baseball":0,"sword":0,"sword_exercise":0,"talk":0,"throw":0,"turn":0,"walk":0,"wave":0}
            
                Q = deque(maxlen=n_frames_p_video)
                R = deque(maxlen=n_frames_p_video)
                S = deque(maxlen=n_frames_p_video)
                frames_tomados = 0
                for ruta_frame in ruta_frames:
                    if frames_tomados == n_frames_p_video:
                        results1 = np.array(Q).mean(axis=0)
                        results2 = np.array(R).mean(axis=0)
                        results3 = np.array(S).mean(axis=0)
                        i1 = np.argmax(results1)
                        i2 = np.argmax(results2)
                        i3 = np.argmax(results3)
                        label1 = lb.classes_[i1]
                        label2 = lb.classes_[i2]
                        label3 = lb.classes_[i3]
                        labels_models = [label1,label2,label3]
                        predictions = dict()
                        for label in labels_models:
                            if label not in predictions.keys():
                                predictions.update({label:1})
                            else:
                                predictions.update({label:predictions.get(label)+1})
            
                        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
                        true_label = sorted_predictions[0][0]
                        clases[true_label] += 1
            
                        frames_tomados = 0
                        Q = deque(maxlen=n_frames_p_video)
                        R = deque(maxlen=n_frames_p_video)
                        S = deque(maxlen=n_frames_p_video)
            
                    else:
                        frame = cv2.imread(ruta_frame,-1)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame1 = preprocessing(best_5_architectures[0],frame)
                        frame2 = preprocessing(best_5_architectures[1],frame)
                        frame3 = preprocessing(best_5_architectures[2],frame)
            
            
                        preds1 = model1.predict(np.expand_dims(frame1, axis=0))[0]
                        preds2 = model2.predict(np.expand_dims(frame2, axis=0))[0]
                        preds3 = model3.predict(np.expand_dims(frame3, axis=0))[0]
                        Q.append(preds1)
                        R.append(preds2)
                        S.append(preds3)
            
                        frames_tomados += 1
            
                total += clases[clase]
                print("El numero de videos de la clase %s es de %d" %(str(clase),len(ruta_frames)/n_frames_p_video))  
                print("El numero de videos que fue predecido como clase %s es de %d" %(str(clase),clases[clase]))
                print("El numero de videos que fue predecido incorrectamente es de %d" %(len(ruta_frames)/n_frames_p_video-clases[clase]))
                print("\n")
            print("Numero total de aciertos: %d" %(total))
            print("Precision: %f" %((total/1530)*100))           
        else:
            model1 = load_model(best_5_models[0])
            model2 = load_model(best_5_models[1])
            model3 = load_model(best_5_models[2])
            model4 = load_model(best_5_models[3])
            model5 = load_model(best_5_models[4])
            total = 0
            for clase in labels:
                etiqueta = path_frame_dataset + str(clase)
                ruta_frames = ls(ruta=etiqueta)
                clases = {"brush_hair":0,"cartwheel":0,"catch":0,"chew":0,"clap":0,"climb":0,"climb_stairs":0,
                "dive":0,"draw_sword":0,"dribble":0,"drink":0,"eat":0,"fall_floor":0,"fencing":0,"flic_flac":0,
                "golf":0,"handstand":0,"hit":0,"hug":0,"jump":0,"kick":0,"kick_ball":0,"kiss":0,"laugh":0,"pick":0,
                "pour":0,"pullup":0,"punch":0,"push":0,"pushup":0,"ride_bike":0,"ride_horse":0,"run":0,"shake_hands":0,
                "shoot_ball":0,"shoot_bow":0,"shoot_gun":0,"sit":0,"situp":0,"smile":0,"smoke":0,"somersault":0,
                "stand":0,"swing_baseball":0,"sword":0,"sword_exercise":0,"talk":0,"throw":0,"turn":0,"walk":0,"wave":0}
            
                Q = deque(maxlen=n_frames_p_video)
                R = deque(maxlen=n_frames_p_video)
                S = deque(maxlen=n_frames_p_video)
                T = deque(maxlen=n_frames_p_video)
                U = deque(maxlen=n_frames_p_video)
                frames_tomados = 0
                for ruta_frame in ruta_frames:
                    if frames_tomados == n_frames_p_video:
                        results1 = np.array(Q).mean(axis=0)
                        results2 = np.array(R).mean(axis=0)
                        results3 = np.array(S).mean(axis=0)
                        results4 = np.array(T).mean(axis=0)
                        results5 = np.array(U).mean(axis=0)
                        i1 = np.argmax(results1)
                        i2 = np.argmax(results2)
                        i3 = np.argmax(results3)
                        i4 = np.argmax(results4)
                        i5 = np.argmax(results5)
                        label1 = lb.classes_[i1]
                        label2 = lb.classes_[i2]
                        label3 = lb.classes_[i3]
                        label4 = lb.classes_[i4]
                        label5 = lb.classes_[i5]
                        labels_models = [label1,label2,label3,label4,label5]
                        predictions = dict()
                        for label in labels_models:
                            if label not in predictions.keys():
                                predictions.update({label:1})
                            else:
                                predictions.update({label:predictions.get(label)+1})
            
                        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
                        true_label = sorted_predictions[0][0]
                        clases[true_label] += 1
            
                        frames_tomados = 0
                        Q = deque(maxlen=n_frames_p_video)
                        R = deque(maxlen=n_frames_p_video)
                        S = deque(maxlen=n_frames_p_video)
                        T = deque(maxlen=n_frames_p_video)
                        U = deque(maxlen=n_frames_p_video)
            
                    else:
                        frame = cv2.imread(ruta_frame,-1)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame1 = preprocessing(best_5_architectures[0],frame)
                        frame2 = preprocessing(best_5_architectures[1],frame)
                        frame3 = preprocessing(best_5_architectures[2],frame)
                        frame4 = preprocessing(best_5_architectures[3],frame)
                        frame5 = preprocessing(best_5_architectures[4],frame)
            
            
                        preds1 = model1.predict(np.expand_dims(frame1, axis=0))[0]
                        preds2 = model2.predict(np.expand_dims(frame2, axis=0))[0]
                        preds3 = model3.predict(np.expand_dims(frame3, axis=0))[0]
                        preds4 = model4.predict(np.expand_dims(frame4, axis=0))[0]
                        preds5 = model5.predict(np.expand_dims(frame5, axis=0))[0]
                        Q.append(preds1)
                        R.append(preds2)
                        S.append(preds3)
                        T.append(preds4)
                        U.append(preds5)
            
                        frames_tomados += 1
            
                total += clases[clase]
                print("El numero de videos de la clase %s es de %d" %(str(clase),len(ruta_frames)/n_frames_p_video))  
                print("El numero de videos que fue predecido como clase %s es de %d" %(str(clase),clases[clase]))
                print("El numero de videos que fue predecido incorrectamente es de %d" %(len(ruta_frames)/n_frames_p_video-clases[clase]))
                print("\n")
            print("Numero total de aciertos: %d" %(total))
            print("Precision: %f" %((total/1530)*100))            
    elif concensus_function == 1: #weighted voting
        if number_of_models == 3:
            model1 = load_model(best_5_models[0])
            model2 = load_model(best_5_models[1])
            model3 = load_model(best_5_models[2])
            total = 0
            for clase in labels:
                etiqueta = path_frame_dataset + str(clase)
                ruta_frames = ls(ruta=etiqueta)
                clases = {"brush_hair":0,"cartwheel":0,"catch":0,"chew":0,"clap":0,"climb":0,"climb_stairs":0,
                "dive":0,"draw_sword":0,"dribble":0,"drink":0,"eat":0,"fall_floor":0,"fencing":0,"flic_flac":0,
                "golf":0,"handstand":0,"hit":0,"hug":0,"jump":0,"kick":0,"kick_ball":0,"kiss":0,"laugh":0,"pick":0,
                "pour":0,"pullup":0,"punch":0,"push":0,"pushup":0,"ride_bike":0,"ride_horse":0,"run":0,"shake_hands":0,
                "shoot_ball":0,"shoot_bow":0,"shoot_gun":0,"sit":0,"situp":0,"smile":0,"smoke":0,"somersault":0,
                "stand":0,"swing_baseball":0,"sword":0,"sword_exercise":0,"talk":0,"throw":0,"turn":0,"walk":0,"wave":0}
            
                Q = deque(maxlen=n_frames_p_video)
                R = deque(maxlen=n_frames_p_video)
                S = deque(maxlen=n_frames_p_video)
                frames_tomados = 0
                for ruta_frame in ruta_frames:
                    if frames_tomados == n_frames_p_video:
                        results1 = np.array(Q).mean(axis=0)
                        results2 = np.array(R).mean(axis=0)
                        results3 = np.array(S).mean(axis=0)
                        i1 = np.argmax(results1)
                        i2 = np.argmax(results2)
                        i3 = np.argmax(results3)
                        label1 = lb.classes_[i1]
                        label2 = lb.classes_[i2]
                        label3 = lb.classes_[i3]
                        labels_models = [label1,label2,label3]
                        predictions = dict()
                        for i,label in enumerate(labels_models):
                            if i == 0:
                                value = 3
                            elif i == 1:
                                value = 2
                            else:
                                value = 1
                            if label not in predictions.keys():
                                predictions.update({label:value})
                            else:
                                predictions.update({label:predictions.get(label)+value})
            
                        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
                        true_label = sorted_predictions[0][0]
                        clases[true_label] += 1
            
                        frames_tomados = 0
                        Q = deque(maxlen=n_frames_p_video)
                        R = deque(maxlen=n_frames_p_video)
                        S = deque(maxlen=n_frames_p_video)
            
                    else:
                        frame = cv2.imread(ruta_frame,-1)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame1 = preprocessing(best_5_architectures[0],frame)
                        frame2 = preprocessing(best_5_architectures[1],frame)
                        frame3 = preprocessing(best_5_architectures[2],frame)
            
            
                        preds1 = model1.predict(np.expand_dims(frame1, axis=0))[0]
                        preds2 = model2.predict(np.expand_dims(frame2, axis=0))[0]
                        preds3 = model3.predict(np.expand_dims(frame3, axis=0))[0]
                        Q.append(preds1)
                        R.append(preds2)
                        S.append(preds3)
            
                        frames_tomados += 1
            
                total += clases[clase]
                print("El numero de videos de la clase %s es de %d" %(str(clase),len(ruta_frames)/n_frames_p_video))  
                print("El numero de videos que fue predecido como clase %s es de %d" %(str(clase),clases[clase]))
                print("El numero de videos que fue predecido incorrectamente es de %d" %(len(ruta_frames)/n_frames_p_video-clases[clase]))
                print("\n")
            print("Numero total de aciertos: %d" %(total))
            print("Precision: %f" %((total/1530)*100))
        else:
            model1 = load_model(best_5_models[0])
            model2 = load_model(best_5_models[1])
            model3 = load_model(best_5_models[2])
            model4 = load_model(best_5_models[3])
            model5 = load_model(best_5_models[4])
            total = 0
            for clase in labels:
                etiqueta = path_frame_dataset + str(clase)
                ruta_frames = ls(ruta=etiqueta)
                clases = {"brush_hair":0,"cartwheel":0,"catch":0,"chew":0,"clap":0,"climb":0,"climb_stairs":0,
                "dive":0,"draw_sword":0,"dribble":0,"drink":0,"eat":0,"fall_floor":0,"fencing":0,"flic_flac":0,
                "golf":0,"handstand":0,"hit":0,"hug":0,"jump":0,"kick":0,"kick_ball":0,"kiss":0,"laugh":0,"pick":0,
                "pour":0,"pullup":0,"punch":0,"push":0,"pushup":0,"ride_bike":0,"ride_horse":0,"run":0,"shake_hands":0,
                "shoot_ball":0,"shoot_bow":0,"shoot_gun":0,"sit":0,"situp":0,"smile":0,"smoke":0,"somersault":0,
                "stand":0,"swing_baseball":0,"sword":0,"sword_exercise":0,"talk":0,"throw":0,"turn":0,"walk":0,"wave":0}
            
                Q = deque(maxlen=20)
                R = deque(maxlen=20)
                S = deque(maxlen=20)
                T = deque(maxlen=20)
                U = deque(maxlen=20)
                frames_tomados = 0
                for ruta_frame in ruta_frames:
                    if frames_tomados == 20:
                        results1 = np.array(Q).mean(axis=0)
                        results2 = np.array(R).mean(axis=0)
                        results3 = np.array(S).mean(axis=0)
                        results4 = np.array(T).mean(axis=0)
                        results5 = np.array(U).mean(axis=0)
                        i1 = np.argmax(results1)
                        i2 = np.argmax(results2)
                        i3 = np.argmax(results3)
                        i4 = np.argmax(results4)
                        i5 = np.argmax(results5)
                        label1 = lb.classes_[i1]
                        label2 = lb.classes_[i2]
                        label3 = lb.classes_[i3]
                        label4 = lb.classes_[i4]
                        label5 = lb.classes_[i5]
                        labels_models = [label1,label2,label3,label4,label5]
                        predictions = dict()
                        for i,label in enumerate(labels_models):
                            if i == 0:
                                value = 5
                            elif i == 1:
                                value = 4
                            elif i == 2:
                                value = 3
                            elif i == 3:
                                value = 2
                            else:
                                value = 1
                            if label not in predictions.keys():
                                predictions.update({label:value})
                            else:
                                predictions.update({label:predictions.get(label)+value})
            
                        sorted_predictions = sorted(predictions.items(), key=lambda x: x[1], reverse=True)
            
                        true_label = sorted_predictions[0][0]
                        clases[true_label] += 1
            
                        frames_tomados = 0
                        Q = deque(maxlen=20)
                        R = deque(maxlen=20)
                        S = deque(maxlen=20)
                        T = deque(maxlen=20)
                        U = deque(maxlen=20)
            
                    else:
                        frame = cv2.imread(ruta_frame,-1)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame1 = preprocessing(best_5_architectures[0],frame)
                        frame2 = preprocessing(best_5_architectures[1],frame)
                        frame3 = preprocessing(best_5_architectures[2],frame)
                        frame4 = preprocessing(best_5_architectures[3],frame)
                        frame5 = preprocessing(best_5_architectures[4],frame)
            
            
                        preds1 = model1.predict(np.expand_dims(frame1, axis=0))[0]
                        preds2 = model2.predict(np.expand_dims(frame2, axis=0))[0]
                        preds3 = model3.predict(np.expand_dims(frame3, axis=0))[0]
                        preds4 = model4.predict(np.expand_dims(frame4, axis=0))[0]
                        preds5 = model5.predict(np.expand_dims(frame5, axis=0))[0]
                        Q.append(preds1)
                        R.append(preds2)
                        S.append(preds3)
                        T.append(preds4)
                        U.append(preds5)
            
                        frames_tomados += 1
            
                total += clases[clase]
                print("El numero de videos de la clase %s es de %d" %(str(clase),len(ruta_frames)/20))  
                print("El numero de videos que fue predecido como clase %s es de %d" %(str(clase),clases[clase]))
                print("El numero de videos que fue predecido incorrectamente es de %d" %(len(ruta_frames)/20-clases[clase]))
                print("\n")
            print("Numero total de aciertos: %d" %(total))
            print("Precision: %f" %((total/1530)*100))
    else:
        if number_of_models == 3: #average of predictions
            model1 = load_model(best_5_models[0])
            model2 = load_model(best_5_models[1])
            model3 = load_model(best_5_models[2])
            total = 0
            for clase in labels:
                etiqueta = path_frame_dataset + str(clase)
                ruta_frames = ls(ruta=etiqueta)
                clases = {"brush_hair":0,"cartwheel":0,"catch":0,"chew":0,"clap":0,"climb":0,"climb_stairs":0,
                "dive":0,"draw_sword":0,"dribble":0,"drink":0,"eat":0,"fall_floor":0,"fencing":0,"flic_flac":0,
                "golf":0,"handstand":0,"hit":0,"hug":0,"jump":0,"kick":0,"kick_ball":0,"kiss":0,"laugh":0,"pick":0,
                "pour":0,"pullup":0,"punch":0,"push":0,"pushup":0,"ride_bike":0,"ride_horse":0,"run":0,"shake_hands":0,
                "shoot_ball":0,"shoot_bow":0,"shoot_gun":0,"sit":0,"situp":0,"smile":0,"smoke":0,"somersault":0,
                "stand":0,"swing_baseball":0,"sword":0,"sword_exercise":0,"talk":0,"throw":0,"turn":0,"walk":0,"wave":0}
            
                Q = deque(maxlen=n_frames_p_video*3)
                frames_tomados = 0
                for ruta_frame in ruta_frames:
                    if frames_tomados == n_frames_p_video:
                        results = np.array(Q).mean(axis=0)
                        i = np.argmax(results)
                        label = lb.classes_[i]
                        clases[label] += 1
            
                        frames_tomados = 0
                        Q = deque(maxlen=n_frames_p_video*3)
            
                    else:
                        frame = cv2.imread(ruta_frame,-1)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame1 = preprocessing(best_5_architectures[0],frame)
                        frame2 = preprocessing(best_5_architectures[1],frame)
                        frame3 = preprocessing(best_5_architectures[2],frame)
            
            
                        preds1 = model1.predict(np.expand_dims(frame1, axis=0))[0]
                        preds2 = model2.predict(np.expand_dims(frame2, axis=0))[0]
                        preds3 = model3.predict(np.expand_dims(frame3, axis=0))[0]
                        Q.append(preds1)
                        Q.append(preds2)
                        Q.append(preds3)
            
                        frames_tomados += 1
            
                total += clases[clase]
                print("El numero de videos de la clase %s es de %d" %(str(clase),len(ruta_frames)/n_frames_p_video))  
                print("El numero de videos que fue predecido como clase %s es de %d" %(str(clase),clases[clase]))
                print("El numero de videos que fue predecido incorrectamente es de %d" %(len(ruta_frames)/n_frames_p_video-clases[clase]))
                print("\n")
            print("Numero total de aciertos: %d" %(total))
            print("Precision: %f" %((total/1530)*100))
        else:
            model1 = load_model(best_5_models[0])
            model2 = load_model(best_5_models[1])
            model3 = load_model(best_5_models[2])
            model4 = load_model(best_5_models[3])
            model5 = load_model(best_5_models[4])
            total = 0
            for clase in labels:
                etiqueta = path_frame_dataset + str(clase)
                ruta_frames = ls(ruta=etiqueta)
                clases = {"brush_hair":0,"cartwheel":0,"catch":0,"chew":0,"clap":0,"climb":0,"climb_stairs":0,
                "dive":0,"draw_sword":0,"dribble":0,"drink":0,"eat":0,"fall_floor":0,"fencing":0,"flic_flac":0,
                "golf":0,"handstand":0,"hit":0,"hug":0,"jump":0,"kick":0,"kick_ball":0,"kiss":0,"laugh":0,"pick":0,
                "pour":0,"pullup":0,"punch":0,"push":0,"pushup":0,"ride_bike":0,"ride_horse":0,"run":0,"shake_hands":0,
                "shoot_ball":0,"shoot_bow":0,"shoot_gun":0,"sit":0,"situp":0,"smile":0,"smoke":0,"somersault":0,
                "stand":0,"swing_baseball":0,"sword":0,"sword_exercise":0,"talk":0,"throw":0,"turn":0,"walk":0,"wave":0}
            
                Q = deque(maxlen=n_frames_p_video*5)
                frames_tomados = 0
                for ruta_frame in ruta_frames:
                    if frames_tomados == n_frames_p_video:
                        results = np.array(Q).mean(axis=0)
                        i = np.argmax(results)
                        label = lb.classes_[i]
                        clases[label] += 1
            
                        frames_tomados = 0
                        Q = deque(maxlen=n_frames_p_video*5)
            
                    else:
                        frame = cv2.imread(ruta_frame,-1)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frame1 = preprocessing(best_5_architectures[0],frame)
                        frame2 = preprocessing(best_5_architectures[1],frame)
                        frame3 = preprocessing(best_5_architectures[2],frame)
                        frame4 = preprocessing(best_5_architectures[3],frame)
                        frame5 = preprocessing(best_5_architectures[4],frame)
            
            
                        preds1 = model1.predict(np.expand_dims(frame1, axis=0))[0]
                        preds2 = model2.predict(np.expand_dims(frame2, axis=0))[0]
                        preds3 = model3.predict(np.expand_dims(frame3, axis=0))[0]
                        preds4 = model4.predict(np.expand_dims(frame4, axis=0))[0]
                        preds5 = model5.predict(np.expand_dims(frame5, axis=0))[0]
                        Q.append(preds1)
                        Q.append(preds2)
                        Q.append(preds3)
                        Q.append(preds4)
                        Q.append(preds5)
            
                        frames_tomados += 1
            
                total += clases[clase]
                print("El numero de videos de la clase %s es de %d" %(str(clase),len(ruta_frames)/n_frames_p_video))  
                print("El numero de videos que fue predecido como clase %s es de %d" %(str(clase),clases[clase]))
                print("El numero de videos que fue predecido incorrectamente es de %d" %(len(ruta_frames)/n_frames_p_video-clases[clase]))
                print("\n")
            print("Numero total de aciertos: %d" %(total))
            print("Precision: %f" %((total/1530)*100))    