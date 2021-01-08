# -*- coding: utf-8 -*-
"""
Created on Tue Jan 5 2021 using Spyder on Windows 10

@author: David Silva
"""


"""This program creates K Fold sets for CV from the sets of the HMDB51 video 
dataset. 

The k variable is the parameter that controls how many sets for CV the
user wants. 

The ruta_origen variable is the path of the folder that contains the training 
videos of any of the 3 sets of the HMDB51 datasets that the user will use to 
create the K Folds sets. The number of folders inside the path folder need to 
be equal to the number of classes in the HMDB51 dataset which is 51. And the 
name of the folders needs to be equal to the name of the HMDB51 classes. 
Example for Ubuntu: ruta_origen = '/home/User/HMDB51_set1_training_videos/' 
Example for Windows: ruta_origen = 'D:/User/HMDB51_set1_training_videos/'

The ruta_destino variable is the path of the folder where the user will put the
created videos for Training and Validation from every K Fold set. Inside of
this destination path folder must be the same number of folders as the value of
the k variable. The name of these folders must be Fold followed by a number. 
For example, if the user wants to create 3 sets for CV, the folders names will 
be 'Fold1', 'Fold2' and 'Fold3'. Inside of every Fold folder must be two 
subfolders with the names 'Train' and 'Val'. And also inside of these 2 
subfolders, there should be 51 folders with the name of the classes of the 
HMDB51 dataset.
Example for Ubuntu: ruta_destino = '/home/User/HMDB51_set1_folds/' 
Example for Windows: ruta_destino = 'D:/User/HMDB51_set1_folds/' """

import random
import shutil
from os import scandir
from os.path import isdir, isfile

#Variables
k  = 3
ruta_origen = ""
ruta_destino = ""

#Constant
LABELS = set(["brush_hair","cartwheel","catch","chew","clap","climb",
    "climb_stairs","dive","draw_sword","dribble","drink","eat","fall_floor",
    "fencing","flic_flac","golf","handstand","hit","hug","jump","kick",
    "kick_ball","kiss","laugh","pick","pour","pullup","punch","push","pushup",
    "ride_bike","ride_horse","run","shake_hands","shoot_ball","shoot_bow",
    "shoot_gun","sit","situp","smile","smoke","somersault",	"stand",
    "swing_baseball","sword","sword_exercise","talk","throw","turn","walk",
    "wave"])


def lsdir(ruta):
    return [arch.name for arch in scandir(ruta) if isdir(arch)]

def lsarch(ruta):
    return [arch.name for arch in scandir(ruta) if isfile(arch)]


k_Folds = [i for i in range(1,k+1)]
for k in k_Folds:
    ruta_final_train = ruta_destino + "Fold" + str(k) + "/Train/"
    ruta_final_val = ruta_destino + "Fold" + str(k) + "/Val/"
    print("[INFO] Creating training and val images for fold %d..." %(k))
    print("\n")
    carpetas = lsdir(ruta_origen)
    for carpeta in carpetas:
        nombres_videos = []
        if carpeta not in LABELS:
            continue
        ruta_final_carpeta_train = ruta_final_train + carpeta + "/"
        ruta_final_carpeta_val = ruta_final_val + carpeta + "/"
        ruta_archivos = ruta_origen + carpeta + "/"
        archivos = lsarch(ruta_archivos)
        for archivo in archivos:
            nombres_videos.append(archivo)
        random.shuffle(nombres_videos)
        videos_restantes = len(nombres_videos)
        while videos_restantes > 21:
            nombre_video = nombres_videos[0]
            ruta_video_origen = ruta_archivos + nombre_video
            ruta_video_destino_Train = ruta_final_carpeta_train + nombre_video
            shutil.copy(ruta_video_origen,ruta_video_destino_Train)
            nombres_videos.pop(0)
            videos_restantes = len(nombres_videos)
        for nombre_video in nombres_videos:
            ruta_video_origen = ruta_archivos + nombre_video
            ruta_video_destino_Val = ruta_final_carpeta_val + nombre_video
            shutil.copy(ruta_video_origen,ruta_video_destino_Val)
            