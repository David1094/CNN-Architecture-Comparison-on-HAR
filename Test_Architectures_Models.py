# -*- coding: utf-8 -*-
"""
Created on Tue Jan 5 2021 using Spyder on Windows 10

@author: David Silva
"""


"""This program loads a pretrained model of a CNN arhitecture and test it
using the frames of the test videos of the set of the HMDB51 video dataset
that was used during training

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

The model_name variable refers to the name of the h5 model that will be load
for testing.

The path_modelo variable refers to the path where the model is saved.
Example for Ubuntu: path = '/home/User/HMDB51_save_models/' 
Example for Windows: path = 'D:/User/HMDB51_save_models/'

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
all_frames = 1
n_frames_p_video = 20
red = ""
model_name = ""
path_modelo = ""
path_frame_dataset = ""
path_video_dataset = ""

#Constants
modelo = path_modelo + model_name
lbpickle = path_modelo + "lb.pickle"


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


if all_frames == 0:
    # load the trained model and label binarizer from disk
    print("[INFO] loading model and label binarizer...")
    model = load_model(filepath=modelo)
    lb = pickle.loads(open(lbpickle, "rb").read())
    
    labels = ["brush_hair","cartwheel","catch","chew","clap","climb","climb_stairs",
    		"dive","draw_sword","dribble","drink","eat","fall_floor","fencing","flic_flac",
    		"golf","handstand","hit","hug","jump","kick","kick_ball","kiss","laugh","pick",
    		"pour","pullup","punch","push","pushup","ride_bike","ride_horse","run","shake_hands",
    		"shoot_ball","shoot_bow","shoot_gun","sit","situp","smile","smoke","somersault",
    		"stand","swing_baseball","sword","sword_exercise","talk","throw","turn","walk","wave"]
    
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
    	frames_tomados = 0
    	for ruta_frame in ruta_frames:
    		if frames_tomados == n_frames_p_video:
    			results = np.array(Q).mean(axis=0)
    			i = np.argmax(results)
    			label = lb.classes_[i]
    			clases[label] += 1
    
    			frames_tomados = 0
    			Q = deque(maxlen=n_frames_p_video)
    
    		else:
    			frame = cv2.imread(ruta_frame,-1)
    			frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    			frame = preprocessing(red,frame)
    
    
    			preds = model.predict(np.expand_dims(frame, axis=0))[0]
    			Q.append(preds)
    
    			frames_tomados += 1
    	
    	total += clases[clase]
    	print("El numero de videos de la clase %s es de %d" %(str(clase),len(ruta_frames)/n_frames_p_video))	
    	print("El numero de videos que fue predecido como clase %s es de %d" %(str(clase),clases[clase]))
    	print("El numero de videos que fue predecido incorrectamente es de %d" %(len(ruta_frames)/n_frames_p_video-clases[clase]))
    	print("\n")
    print("Numero total de aciertos: %d" %(total))
    print("Precision: %f" %((total/((len(ruta_frames)/n_frames_p_video)*51))*100))
else:
    # load the trained model and label binarizer from disk
    print("[INFO] loading model and label binarizer...")
    model = load_model(filepath=modelo)
    lb = pickle.loads(open(lbpickle, "rb").read())
    
    labels = ["brush_hair","cartwheel","catch","chew","clap","climb","climb_stairs",
        		"dive","draw_sword","dribble","drink","eat","fall_floor","fencing","flic_flac",
        		"golf","handstand","hit","hug","jump","kick","kick_ball","kiss","laugh","pick",
        		"pour","pullup","punch","push","pushup","ride_bike","ride_horse","run","shake_hands",
        		"shoot_ball","shoot_bow","shoot_gun","sit","situp","smile","smoke","somersault",
        		"stand","swing_baseball","sword","sword_exercise","talk","throw","turn","walk","wave"]
    
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
    
            vs2 = cv2.VideoCapture(videos[clip])
    
            while(True):
                (grabbed, frame) = vs2.read()
    
                if not grabbed:
                    break
    
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = preprocessing(red,frame)
    
                preds = model.predict(np.expand_dims(frame, axis=0))[0]
                Q.append(preds)
    
            results = np.array(Q).mean(axis=0)
            i = np.argmax(results)
            label = lb.classes_[i]
            clases[label] += 1
            # release the file pointers
            vs.release()
            vs2.release()
        total += clases[clase]
        print("El numero de videos de la clase %s es de %d" %(str(clase),len(videos)))    
        print("El numero de videos que fue predecido como clase %s es de %d" %(str(clase),clases[clase]))
        print("El numero de videos que fue predecido incorrectamente es de %d" %(len(videos)-clases[clase]))
        print("\n")
    print("Numero total de aciertos: %d" %(total))
    print("Precision: %f" %((total/1530)*100))