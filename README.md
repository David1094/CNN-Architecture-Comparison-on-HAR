# CNN-Architecture-Comparison-on-HAR
CNN Architecture Comparison on Human Action Recognition using keras and HMDB51 dataset.

This repository holds the code of the tesis named "Analisis de Arquitecturas CNN del estado arte en el reconocimiento de actividades en video" made by M.C.I. David Silva
NOTE: All the programs must be edited before using them, all of them have different variables and all variables have a descripcion inside the program so make sure to read them.

STEP 1: You need to download the HMDB51 file and the three splits for the HMDB51 file of this link https://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/.
STEP 2: After you have downloaded both files, you need to use the program Make_Train_Test_Splits.py to make the 3 Test Train Splits of HMDB51.
STEP 3: After creating the 3 train-test splits you have two options. You can either use the program Create_K_Fold_Video_Sets_for_CV.py to create k extra sets of videos for cross validation using the videos of one of the three sets as a reference and after that you can extract the frames of those sets using the program called Create_Dataset_of_Frames.py or you can begin to extract the frames of the videos in each set using the program called Create_Dataset_of_Frames.py
STEP 4: After the frames are extracted you can train the CNN Architectures with them using the program Train_Architectures.py.
STEP 5: You can test the accuracy of the models based on the CNN Architectures using the Test_Architectures_Models.py program.
STEP 6: You can combine the models that you trained with the same dataset in an ensemble using the Test_Ensembles.py program.

I also attached 2 extra programs. The one that is called Optimizer_Selection_Using_Shallow_CNN.py works like the Train_Architectures.py program but it uses the frames extracted of the K Folds CV sets for optimizer selection and also uses a shallow CNN proposed by Francois Chollet at this link https://blog.keras.io/building-powerful-image-classification-models-using-very-little-data.html for faster training.

The other file named Scene_Segmentation.py is a program that I used to try to clean a little the frames that appear in the videos of the HMDB51 dataset. It first goes through all the frames of all the videos in the HMDB51 dataset and if all the pixels intensities of a frame fall behind or fall above a certain limit then that frame is remove from the video. After that it uses scene segmentation to classify the videos that have only one scene from the ones that have multiple scenes. I extracted the frames from the videos with one scene and used them to train a CNN Architecture with the goal of predicting which frames of the videos with multiple scenes in a single class are more likely to belong to the class in consideration.


