# -*- coding: utf-8 -*-
"""
Created on Tue Jan 5 2021 using Spyder on Windows 10

@author: David Silva
"""


"""This program creates an image dataset with the frames of the training and
testing videos of the sets of the HMDB51 dataset for later use by a CNN model.

The data_aug variable specifies wheter to use augmentation tecniques while
extracting the frames or not. The values of this variables can be either 0 for
no augmentation at all, 1 for horizontal flipping, 2 for scalling the smallest
size of the image to 256 pixels and crop 5 regions of 224x224 of the scaled 
image (4 corners and 1 center) and 3 is for scalling the smallest size of the 
image to 256 pixels and crop 5 regions of 224x224 of the scaled image 
(4 corners and 1 center) along with the horizontal flip images of these 5 crop 
regions.

The n_train_videos and n_test_videos variables are only useful to see the
progress of the frame extraction process. They refer to the total number of
videos on the train and test folder respectively. They can also be use to
specify the number of training and validation videos when extracting the frames
of every video of a Fold set, in which case the n_test_videos act as the number
of videos of the validation set.

The frames_max_por_video variable refers to the amount of frames that will be
extracted from each video.

The ruta_origen_Train variable is the one that specifies the folder where the 
train videos of a certain set are located in the computer or the train videos
of a certain fold. The folder must contain the 51 folders related to the 
classes of the HMDB51 dataset. Each of the 51 folders must contain the videos 
of that class.
Example for Ubuntu: path = '/home/User/HMDB51_dataset/set1/Train' 
Example for Windows: path = 'D:/User/HMDB51_dataset/set1/Train'
Example for Ubuntu: path = '/home/User/HMDB51_dataset/Fold1/Train' 
Example for Windows: path = 'D:/User/HMDB51_dataset/Fold1/Train'

The ruta_origen_Test variable is the one that specifies the folder where the 
test videos of a certain set are located in the computer or the validation 
videos of a certain fold. The folder must contain the 51 folders related to the 
classes of the HMDB51 dataset. Each of the 51 folders must contain the videos 
of that class.
Example for Ubuntu: path = '/home/User/HMDB51_dataset/set1/Test' 
Example for Windows: path = 'D:/User/HMDB51_dataset/set1/Test'
Example for Ubuntu: path = '/home/User/HMDB51_dataset/Fold1/Val' 
Example for Windows: path = 'D:/User/HMDB51_dataset/Fold1/Val'

The ruta_destino_Train variable is the path of the folder where the user will 
put the generated frames for Training. The folder must contain the 51 folders 
related to the classes of the HMDB51 dataset.
Example for Ubuntu: path = '/home/User/HMDB51_frame_dataset/set1/Train' 
Example for Windows: path = 'D:/User/HMDB51_frame_dataset/set1/Train'
Example for Ubuntu: path = '/home/User/HMDB51_frame_dataset/Fold1/Train' 
Example for Windows: path = 'D:/User/HMDB51_frame_dataset/Fold1/Train'

The ruta_destino_Train variable is the path of the folder where the user will 
put the generated frames for Testing of Validation. The folder must contain 
the 51 folders related to the classes of the HMDB51 dataset.
Example for Ubuntu: path = '/home/User/HMDB51_frame_dataset/set1/Test' 
Example for Windows: path = 'D:/User/HMDB51_frame_dataset/set1/Test'
Example for Ubuntu: path = '/home/User/HMDB51_frame_dataset/Fold1/Val' 
Example for Windows: path = 'D:/User/HMDB51_frame_dataset/Fold1/Val'
"""

from os.path import isdir, isfile
from os import scandir
import cv2 as cv
import math

#Variables
data_aug = 0
n_train_videos = 3570
n_test_videos = 1530
frames_max_por_video = 10
ruta_origen_Train = ""
ruta_origen_Test = ""
ruta_destino_Train = ""
ruta_destino_Test = ""

#Constant
LABELS = set(["brush_hair","cartwheel","catch","chew","clap","climb",
    "climb_stairs","dive","draw_sword","dribble","drink","eat","fall_floor",
    "fencing","flic_flac","golf","handstand","hit","hug","jump","kick",
    "kick_ball","kiss","laugh","pick","pour","pullup","punch","push","pushup",
    "ride_bike","ride_horse","run","shake_hands","shoot_ball","shoot_bow",
    "shoot_gun","sit","situp","smile","smoke","somersault","stand",
    "swing_baseball","sword","sword_exercise","talk","throw","turn","walk",
    "wave"])


def lsdir(ruta):
    return [arch.name for arch in scandir(ruta) if isdir(arch)]

def lsarch(ruta):
    return [arch.name for arch in scandir(ruta) if isfile(arch)]


if data_aug == 0:
    print("\n[INFO] Saving train images...")
    carpetas = lsdir(ruta_origen_Train)
    videos_totales = 0
    for carpeta in carpetas:
        video_carpeta = 0
        if carpeta not in LABELS:
            continue
        ruta_archivos = ruta_origen_Train + '/' + carpeta
        archivos = lsarch(ruta_archivos)
        for archivo in archivos:
            videos_totales += 1
            video_carpeta += 1
            ruta_video = ruta_archivos + '/' + archivo
            video = cv.VideoCapture(ruta_video)
            frame_actual = 0
            n_frames_video = 0
            total_frames_extraidos = 0
            while(True):
                (grabbed, image) = video.read()
                if not grabbed:
                    break
                n_frames_video += 1
            espacio_frames = n_frames_video // frames_max_por_video
            video.release()
            video = cv.VideoCapture(ruta_video)
            while(True):
                (grabbed, image) = video.read()
                if not grabbed or total_frames_extraidos == frames_max_por_video:
                    break
                if frame_actual % espacio_frames == 0 and total_frames_extraidos < frames_max_por_video:
                    total_frames_extraidos += 1
                    ruta_destino_final = ruta_destino_Train + "/" + carpeta + "/" + carpeta + "_video_" + str(video_carpeta) + "_frame_" + str(frame_actual) + ".png"
                    cv.imwrite(ruta_destino_final,cv.resize(image,(224,224)))
                frame_actual += 1            
            video.release()  
            print("%6.2f %%" %((videos_totales/n_train_videos)*100), end='\r')
    print("\n[INFO] Saving test images...")
    carpetas = lsdir(ruta_origen_Test)
    videos_totales = 0
    for carpeta in carpetas:
        video_carpeta = 0
        if carpeta not in LABELS:
            continue
        ruta_archivos = ruta_origen_Test + '/' + carpeta
        archivos = lsarch(ruta_archivos)
        for archivo in archivos:
            videos_totales += 1
            video_carpeta += 1
            ruta_video = ruta_archivos + '/' + archivo
            video = cv.VideoCapture(ruta_video)
            frame_actual = 0
            n_frames_video = 0
            total_frames_extraidos = 0
            while(True):
                (grabbed, image) = video.read()
                if not grabbed:
                    break
                n_frames_video += 1
            espacio_frames = n_frames_video // frames_max_por_video
            video.release()
            video = cv.VideoCapture(ruta_video)
            while(True):
                (grabbed, image) = video.read()
                if not grabbed or total_frames_extraidos == frames_max_por_video:
                    break
                if frame_actual % espacio_frames == 0 and total_frames_extraidos < frames_max_por_video:
                    total_frames_extraidos += 1
                    ruta_destino_final = ruta_destino_Test + "/" + carpeta + "/" + carpeta + "_video_" + str(video_carpeta) + "_frame_" + str(frame_actual) + ".png"
                    cv.imwrite(ruta_destino_final,cv.resize(image,(224,224)))
                frame_actual += 1            
            video.release()  
            print("%6.2f %%" %((videos_totales/n_test_videos)*100), end='\r')
elif data_aug == 1:
    print("\n[INFO] Saving train images...")
    carpetas = lsdir(ruta_origen_Train)
    videos_totales = 0
    for carpeta in carpetas:
        video_carpeta = 0
        if carpeta not in LABELS:
            continue
        ruta_archivos = ruta_origen_Train + '/' + carpeta
        archivos = lsarch(ruta_archivos)
        for archivo in archivos:
            videos_totales += 1
            video_carpeta += 1
            ruta_video = ruta_archivos + '/' + archivo
            video = cv.VideoCapture(ruta_video)
            frame_actual = 0
            n_frames_video = 0
            total_frames_extraidos = 0
            while(True):
                (grabbed, image) = video.read()
                if not grabbed:
                    break
                n_frames_video += 1
            espacio_frames = n_frames_video // frames_max_por_video
            video.release()
            video = cv.VideoCapture(ruta_video)
            while(True):
                (grabbed, image) = video.read()
                if not grabbed or total_frames_extraidos == frames_max_por_video:
                    break
                if frame_actual % espacio_frames == 0 and total_frames_extraidos < frames_max_por_video:
                    total_frames_extraidos += 1
                    flipped = cv.flip(image,1)
                    ruta_destino_final = ruta_destino_Train + "/" + carpeta + "/" + carpeta + "_video_" + str(video_carpeta) + "_frame_" + str(frame_actual)
                    cv.imwrite(ruta_destino_final + "_Imagen1.png",cv.resize(image,(224,224)))
                    cv.imwrite(ruta_destino_final + "_Imagen2.png",cv.resize(flipped,(224,224))) 
                frame_actual += 1            
            video.release()  
            print("%6.2f %%" %((videos_totales/n_train_videos)*100), end='\r')
    print("\n[INFO] Saving test images...")
    carpetas = lsdir(ruta_origen_Test)
    videos_totales = 0
    for carpeta in carpetas:
        video_carpeta = 0
        if carpeta not in LABELS:
            continue
        ruta_archivos = ruta_origen_Test + '/' + carpeta
        archivos = lsarch(ruta_archivos)
        for archivo in archivos:
            videos_totales += 1
            video_carpeta += 1
            ruta_video = ruta_archivos + '/' + archivo
            video = cv.VideoCapture(ruta_video)
            frame_actual = 0
            n_frames_video = 0
            total_frames_extraidos = 0
            while(True):
                (grabbed, image) = video.read()
                if not grabbed:
                    break
                n_frames_video += 1
            espacio_frames = n_frames_video // frames_max_por_video
            video.release()
            video = cv.VideoCapture(ruta_video)
            while(True):
                (grabbed, image) = video.read()
                if not grabbed or total_frames_extraidos == frames_max_por_video:
                    break
                if frame_actual % espacio_frames == 0 and total_frames_extraidos < frames_max_por_video:
                    total_frames_extraidos += 1
                    flipped = cv.flip(image,1)
                    ruta_destino_final = ruta_destino_Test + "/" + carpeta + "/" + carpeta + "_video_" + str(video_carpeta) + "_frame_" + str(frame_actual)
                    cv.imwrite(ruta_destino_final + "_Imagen1.png",cv.resize(image,(224,224)))
                    cv.imwrite(ruta_destino_final + "_Imagen2.png",cv.resize(flipped,(224,224))) 
                frame_actual += 1            
            video.release()  
            print("%6.2f %%" %((videos_totales/n_test_videos)*100), end='\r')
elif data_aug == 2:
    print("\n[INFO] Saving train images...")
    carpetas = lsdir(ruta_origen_Train)
    videos_totales = 0
    for carpeta in carpetas:
        video_carpeta = 0
        if carpeta not in LABELS:
            continue
        ruta_archivos = ruta_origen_Train + '/' + carpeta
        archivos = lsarch(ruta_archivos)
        for archivo in archivos:
            videos_totales += 1
            video_carpeta += 1
            ruta_video = ruta_archivos + '/' + archivo
            video = cv.VideoCapture(ruta_video)
            frame_actual = 0
            n_frames_video = 0
            total_frames_extraidos = 0
            while(True):
                (grabbed, image) = video.read()
                if not grabbed:
                    break
                n_frames_video += 1
            espacio_frames = n_frames_video // frames_max_por_video
            video.release()
            video = cv.VideoCapture(ruta_video)
            while(True):
                (grabbed, image) = video.read()
                if not grabbed or total_frames_extraidos == frames_max_por_video:
                    break
                if frame_actual % espacio_frames == 0 and total_frames_extraidos < frames_max_por_video:
                    total_frames_extraidos += 1
                    height,width,_ = image.shape

                    if width>height:
                        escala=256/height
                    else:
                        escala=256/width
                        
                    new_width = int(round(width * escala))
                    new_height = int(round(height * escala))
                        
                    imagen = cv.resize(image, (new_width,new_height), interpolation=cv.INTER_AREA)
                    flipped = cv.flip(imagen,1)
                    h,w,_ = imagen.shape
                    ruta_destino_final = ruta_destino_Train + "/" + carpeta + "/" + carpeta + "_video_" + str(video_carpeta) + "_frame_" + str(frame_actual)
                    cv.imwrite(ruta_destino_final+"_Imagen1.jpg",imagen[0:224,0:224,:])
                    cv.imwrite(ruta_destino_final+"_Imagen2.jpg",imagen[0:224,w-224:w,:])
                    cv.imwrite(ruta_destino_final+"_Imagen3.jpg",imagen[math.floor((h-224)/2):h-math.ceil((h-224)/2),math.floor((w-224)/2):w-math.ceil((w-224)/2),:])
                    cv.imwrite(ruta_destino_final+"_Imagen4.jpg",imagen[h-224:h,0:224,:])
                    cv.imwrite(ruta_destino_final+"_Imagen5.jpg",imagen[h-224:h,w-224:w,:])
                frame_actual += 1            
            video.release()  
            print("%6.2f %%" %((videos_totales/n_train_videos)*100), end='\r')
    print("\n[INFO] Saving test images...")
    carpetas = lsdir(ruta_origen_Test)
    videos_totales = 0
    for carpeta in carpetas:
        video_carpeta = 0
        if carpeta not in LABELS:
            continue
        ruta_archivos = ruta_origen_Test + '/' + carpeta
        archivos = lsarch(ruta_archivos)
        for archivo in archivos:
            videos_totales += 1
            video_carpeta += 1
            ruta_video = ruta_archivos + '/' + archivo
            video = cv.VideoCapture(ruta_video)
            frame_actual = 0
            n_frames_video = 0
            total_frames_extraidos = 0
            while(True):
                (grabbed, image) = video.read()
                if not grabbed:
                    break
                n_frames_video += 1
            espacio_frames = n_frames_video // frames_max_por_video
            video.release()
            video = cv.VideoCapture(ruta_video)
            while(True):
                (grabbed, image) = video.read()
                if not grabbed or total_frames_extraidos == frames_max_por_video:
                    break
                if frame_actual % espacio_frames == 0 and total_frames_extraidos < frames_max_por_video:
                    total_frames_extraidos += 1
                    height,width,_ = image.shape

                    if width>height:
                        escala=256/height
                    else:
                        escala=256/width
                        
                    new_width = int(round(width * escala))
                    new_height = int(round(height * escala))
                        
                    imagen = cv.resize(image, (new_width,new_height), interpolation=cv.INTER_AREA)
                    flipped = cv.flip(imagen,1)
                    h,w,_ = imagen.shape
                    ruta_destino_final = ruta_destino_Test + "/" + carpeta + "/" + carpeta + "_video_" + str(video_carpeta) + "_frame_" + str(frame_actual)
                    cv.imwrite(ruta_destino_final+"_Imagen1.jpg",imagen[0:224,0:224,:])
                    cv.imwrite(ruta_destino_final+"_Imagen2.jpg",imagen[0:224,w-224:w,:])
                    cv.imwrite(ruta_destino_final+"_Imagen3.jpg",imagen[math.floor((h-224)/2):h-math.ceil((h-224)/2),math.floor((w-224)/2):w-math.ceil((w-224)/2),:])
                    cv.imwrite(ruta_destino_final+"_Imagen4.jpg",imagen[h-224:h,0:224,:])
                    cv.imwrite(ruta_destino_final+"_Imagen5.jpg",imagen[h-224:h,w-224:w,:])
                frame_actual += 1            
            video.release()  
            print("%6.2f %%" %((videos_totales/n_test_videos)*100), end='\r')
else:
    print("\n[INFO] Saving train images...")
    carpetas = lsdir(ruta_origen_Train)
    videos_totales = 0
    for carpeta in carpetas:
        video_carpeta = 0
        if carpeta not in LABELS:
            continue
        ruta_archivos = ruta_origen_Train + '/' + carpeta
        archivos = lsarch(ruta_archivos)
        for archivo in archivos:
            videos_totales += 1
            video_carpeta += 1
            ruta_video = ruta_archivos + '/' + archivo
            video = cv.VideoCapture(ruta_video)
            frame_actual = 0
            n_frames_video = 0
            total_frames_extraidos = 0
            while(True):
                (grabbed, image) = video.read()
                if not grabbed:
                    break
                n_frames_video += 1
            espacio_frames = n_frames_video // frames_max_por_video
            video.release()
            video = cv.VideoCapture(ruta_video)
            while(True):
                (grabbed, image) = video.read()
                if not grabbed or total_frames_extraidos == frames_max_por_video:
                    break
                if frame_actual % espacio_frames == 0 and total_frames_extraidos < frames_max_por_video:
                    total_frames_extraidos += 1
                    height,width,_ = image.shape

                    if width>height:
                        escala=256/height
                    else:
                        escala=256/width
                        
                    new_width = int(round(width * escala))
                    new_height = int(round(height * escala))
                        
                    imagen = cv.resize(image, (new_width,new_height), interpolation=cv.INTER_AREA)
                    flipped = cv.flip(imagen,1)
                    h,w,_ = imagen.shape
                    ruta_destino_final = ruta_destino_Train + "/" + carpeta + "/" + carpeta + "_video_" + str(video_carpeta) + "_frame_" + str(frame_actual)
                    cv.imwrite(ruta_destino_final+"_Imagen1.jpg",imagen[0:224,0:224,:])
                    cv.imwrite(ruta_destino_final+"_Imagen2.jpg",imagen[0:224,w-224:w,:])
                    cv.imwrite(ruta_destino_final+"_Imagen3.jpg",imagen[math.floor((h-224)/2):h-math.ceil((h-224)/2),math.floor((w-224)/2):w-math.ceil((w-224)/2),:])
                    cv.imwrite(ruta_destino_final+"_Imagen4.jpg",imagen[h-224:h,0:224,:])
                    cv.imwrite(ruta_destino_final+"_Imagen5.jpg",imagen[h-224:h,w-224:w,:])
                    cv.imwrite(ruta_destino_final+"_Imagen6.jpg",flipped[0:224,0:224,:])
                    cv.imwrite(ruta_destino_final+"_Imagen7.jpg",flipped[0:224,w-224:w,:])
                    cv.imwrite(ruta_destino_final+"_Imagen8.jpg",flipped[math.floor((h-224)/2):h-math.ceil((h-224)/2),math.floor((w-224)/2):w-math.ceil((w-224)/2),:])
                    cv.imwrite(ruta_destino_final+"_Imagen9.jpg",flipped[h-224:h,0:224,:])
                    cv.imwrite(ruta_destino_final+"_Imagen10.jpg",flipped[h-224:h,w-224:w,:])  
                frame_actual += 1            
            video.release()  
            print("%6.2f %%" %((videos_totales/n_train_videos)*100), end='\r')
    print("\n[INFO] Saving test images...")
    carpetas = lsdir(ruta_origen_Test)
    videos_totales = 0
    for carpeta in carpetas:
        video_carpeta = 0
        if carpeta not in LABELS:
            continue
        ruta_archivos = ruta_origen_Test + '/' + carpeta
        archivos = lsarch(ruta_archivos)
        for archivo in archivos:
            videos_totales += 1
            video_carpeta += 1
            ruta_video = ruta_archivos + '/' + archivo
            video = cv.VideoCapture(ruta_video)
            frame_actual = 0
            n_frames_video = 0
            total_frames_extraidos = 0
            while(True):
                (grabbed, image) = video.read()
                if not grabbed:
                    break
                n_frames_video += 1
            espacio_frames = n_frames_video // frames_max_por_video
            video.release()
            video = cv.VideoCapture(ruta_video)
            while(True):
                (grabbed, image) = video.read()
                if not grabbed or total_frames_extraidos == frames_max_por_video:
                    break
                if frame_actual % espacio_frames == 0 and total_frames_extraidos < frames_max_por_video:
                    total_frames_extraidos += 1
                    height,width,_ = image.shape

                    if width>height:
                        escala=256/height
                    else:
                        escala=256/width
                        
                    new_width = int(round(width * escala))
                    new_height = int(round(height * escala))
                        
                    imagen = cv.resize(image, (new_width,new_height), interpolation=cv.INTER_AREA)
                    flipped = cv.flip(imagen,1)
                    h,w,_ = imagen.shape
                    ruta_destino_final = ruta_destino_Test + "/" + carpeta + "/" + carpeta + "_video_" + str(video_carpeta) + "_frame_" + str(frame_actual)
                    cv.imwrite(ruta_destino_final+"_Imagen1.jpg",imagen[0:224,0:224,:])
                    cv.imwrite(ruta_destino_final+"_Imagen2.jpg",imagen[0:224,w-224:w,:])
                    cv.imwrite(ruta_destino_final+"_Imagen3.jpg",imagen[math.floor((h-224)/2):h-math.ceil((h-224)/2),math.floor((w-224)/2):w-math.ceil((w-224)/2),:])
                    cv.imwrite(ruta_destino_final+"_Imagen4.jpg",imagen[h-224:h,0:224,:])
                    cv.imwrite(ruta_destino_final+"_Imagen5.jpg",imagen[h-224:h,w-224:w,:])
                    cv.imwrite(ruta_destino_final+"_Imagen6.jpg",flipped[0:224,0:224,:])
                    cv.imwrite(ruta_destino_final+"_Imagen7.jpg",flipped[0:224,w-224:w,:])
                    cv.imwrite(ruta_destino_final+"_Imagen8.jpg",flipped[math.floor((h-224)/2):h-math.ceil((h-224)/2),math.floor((w-224)/2):w-math.ceil((w-224)/2),:])
                    cv.imwrite(ruta_destino_final+"_Imagen9.jpg",flipped[h-224:h,0:224,:])
                    cv.imwrite(ruta_destino_final+"_Imagen10.jpg",flipped[h-224:h,w-224:w,:]) 
                frame_actual += 1            
            video.release()  
            print("%6.2f %%" %((videos_totales/n_test_videos)*100), end='\r')
