# -*- coding: utf-8 -*-
"""
Created on Tue Jan 5 2021 using Spyder on Windows 10

@author: David Silva
"""


"""This is the program that I used to try to optimize the frames in a video in
order to try to get better acc at test time. 

This program is divided in two steps. The first step analyzes all frames of
all videos of the original dataset in search of any frame that have all of 
it pixels beyond or below a certain intensity level. Any black or white frame
will be remove from the video and the remaining frames will be the ones that
the new video will have.

The second step creates two datasets using scene segmentation. Using the
library scenedetect the program can get to every single video of the dataset
that was clean before and check if there is a scene change inside the video.
The second scene must have at least 10 frames to be catalog as an extra scene.
The videos with only one scene will go to one dataset and the videos with 
multiple scene will go to another dataset. The videos from the dataset that 
have only videos with one scene are the ones that I used to train a CNN 
architecture and I used the trained model to try to predict which frames of
the multi-scene videos are more related to the correct class.

The path_original_dataset variable is the one that specifies the folder where 
the HMDB51 video dataset is located in the computer. 
Example for Ubuntu: path = '/home/User/HMDB51/' 
Example for Windows: path = 'D:/User/HMDB51/' 

The path_clean_dataset variable is the one that specifies the folder where the 
videos that were clean from black and white frame will be save. This path 
folder must have the 51 folders corresponding to the names of the HMDB51 
classes.
Example for Ubuntu: path = '/home/User/cleaner_HMDB51/' 
Example for Windows: path = 'D:/User/cleaner_HMDB51/'

The path_videos_one_scene variable is the one that specifies the folder where 
the videos that have only one scene will be save. This path folder must have 
the 51 folders corresponding to the namea of the HMDB51 classes.
Example for Ubuntu: path = '/home/User/HMDB51_single_scene/' 
Example for Windows: path = 'D:/User/HMDB51_single_scene/'

The path_videos_multi_scene variable is the one that specifies the folder where 
the videos that have multiple scenes will be save. This path folder must have 
the 51 folders corresponding to the namea of the HMDB51 classes.
Example for Ubuntu: path = '/home/User/HMDB51_multi_scene/' 
Example for Windows: path = 'D:/User/HMDB51_multi_scene/'
"""

from os import scandir
from os.path import isdir, isfile
import numpy as np
import cv2 as cv
from scenedetect import VideoManager
from scenedetect import SceneManager
import shutil
# For content-aware scene detection:
from scenedetect.detectors import ContentDetector

#Variables
path_original_dataset = ""
path_clean_dataset = ""
path_videos_one_scene = ""
path_videos_multi_scene = ""


def find_scenes(video_path, threshold=30.0):
    # Create our video & scene managers, then add the detector.
    video_manager = VideoManager([video_path])
    scene_manager = SceneManager()
    scene_manager.add_detector(
        ContentDetector(threshold=threshold,min_scene_len=10))

    # Base timestamp at frame 0 (required to obtain the scene list).
    base_timecode = video_manager.get_base_timecode()

    # Improve processing speed by downscaling before processing.
    video_manager.set_downscale_factor()

    # Start the video manager and perform the scene detection.
    video_manager.start()
    scene_manager.detect_scenes(frame_source=video_manager)

    # Each returned scene is a tuple of the (start, end) timecode.
    return scene_manager.get_scene_list(base_timecode)

def lsdir(ruta):
    return [arch.name for arch in scandir(ruta) if isdir(arch)]

def lsarch(ruta):
    return [arch.name for arch in scandir(ruta) if isfile(arch)]

def checkBlackWhiteScreen(frame):
    frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    arreglo_frame = np.ravel(frame)
    return all(x < 10 or x > 245 for x in arreglo_frame)


carpetas = lsdir(path_original_dataset)
for carpeta in carpetas:
    ruta_carpeta = path_original_dataset + "{}/".format(carpeta)
    ruta_carpeta_destino = path_clean_dataset + "{}/".format(carpeta)
    archivos = lsarch(ruta_carpeta)
    for archivo in archivos:
        lista_frames = []
        ruta_video = ruta_carpeta + "{}".format(archivo)
        ruta_video_destino = ruta_carpeta_destino + "{}".format(archivo)
        video = cv.VideoCapture(ruta_video)
        while(True):
            (grabbed, frame) = video.read()
            if not grabbed:
                break
            if not checkBlackWhiteScreen(frame):
                lista_frames.append(frame)
                h,w,_ = frame.shape
        writer = cv.VideoWriter(ruta_video_destino,cv.VideoWriter_fourcc(*'DIVX'), 30, (w,h))
        for frame in lista_frames:  
            writer.write(frame)
        video.release()
        writer.release()


carpetas = lsdir(path_clean_dataset)
for carpeta in carpetas:
    ruta_carpeta = path_clean_dataset + "{}/".format(carpeta)
    ruta_carpeta_destino1 = path_videos_one_scene + "{}/".format(carpeta)
    ruta_carpeta_destino2 = path_videos_multi_scene + "{}/".format(carpeta)
    archivos = lsarch(ruta_carpeta)
    for archivo in archivos:
        ruta_video = ruta_carpeta + "{}".format(archivo)
        ruta_video_destino1 = ruta_carpeta_destino1 + "{}".format(archivo)
        ruta_video_destino2 = ruta_carpeta_destino2 + "{}".format(archivo)
        lista = find_scenes(ruta_video)
        if len(lista) == 1:
            shutil.copy(ruta_video, ruta_video_destino1)
        else:
            shutil.copy(ruta_video, ruta_video_destino2)
            