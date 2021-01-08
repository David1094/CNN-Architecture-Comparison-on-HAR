# -*- coding: utf-8 -*-
"""
Created on Tue Jan 5 2021 using Spyder on Windows 10

@author: David Silva
"""


"""This program create the 3 test-train splits of the HMDB51 video dataset 
using the .txt files that the dataset provides.

The path_videos_origen variable is the one that specifies the folder where the 
HMDB51 video dataset is located in the computer. 
Example for Ubuntu: path = '/home/User/HMDB51/' 
Example for Windows: path = 'D:/User/HMDB51/'

The path_videos_destino variable is the one that specifies the folder where the 
videos of each of the HMDB51 sets will be saved. This folder must have 3 
folders 'TTS1', 'TTS2' and 'TTS3'. Each of these 3 folders must have 2 folders 
'Train' and 'Test'. And any of these 2 folders must contain the 51 folders 
related to the classes of the HMDB51 dataset.
Example for Ubuntu: path = '/home/User/HMDB51_sets/' 
Example for Windows: path = 'D:/User/HMDB51_sets/'

The path_tts variable is the one that specifies the folder where the .txt files
are located.
Example for Ubuntu: path = '/home/User/HMDB51_testTrainMulti_7030_splits/' 
Example for Windows: path = 'D:/User/HMDB51_testTrainMulti_7030_splits/' """


import shutil
from os.path import isfile
from os import scandir

#Variables
path_videos_origen = ""
path_videos_destino = ""
path_tts = ""


def lsarch(ruta):
    return [arch.name for arch in scandir(ruta) if isfile(arch)]


archivos = lsarch(path_tts)
for archivo in archivos:
    split = archivo[-5]
    clase = archivo.split("_test")[0]
    file = open(path_tts+archivo,"r")
    for line in file:
        number = int(line[-3])
        if number == 1:
            tipo = "Train"
        elif number == 2:
            tipo = "Test"
        else:
            continue
        ruta_origen_clases = path_videos_origen + "{}/".format(clase)
        ruta_final = path_videos_destino + "TTS{}/{}/{}".format(split,tipo,clase)
        nombre_video = line.split(" ")[0]
        if nombre_video != "":
            ruta_origen_video = ruta_origen_clases + nombre_video
            shutil.copy(ruta_origen_video, ruta_final)

        
    
