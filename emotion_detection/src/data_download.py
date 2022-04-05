# -*- coding: utf-8 -*-
"""
Created on Wed Apr  6 00:54:56 2022

@author: Lai Tsz Chun
"""


import os
import cv2
import numpy as np
import pandas as pd
import errno
import imageio


# initialization
image_height = 48
image_width = 48
SAVE_IMAGES = True
SELECTED_LABELS = [0,1,2,3,4,5,6]
IMAGES_PER_LABEL = 500
OUTPUT_FOLDER_NAME = "fer2013_features"

def data_download():
    # loading Dlib predictor and preparing arrays:
    print( "preparing")
    #predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
    original_labels = [0, 1, 2, 3, 4, 5, 6]
    new_labels = list(set(original_labels) & set(SELECTED_LABELS))
    nb_images_per_label = list(np.zeros(len(new_labels), 'uint8'))
    try:
        os.makedirs(OUTPUT_FOLDER_NAME)
    except OSError as e:
        if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
            pass
        else:
            raise
            
    print("This file full path (following symlinks)")
    full_path = os.path.realpath("fer2013.csv")    #get your own path
    print(full_path + "\n")      #show your current path

    print( "importing csv file")
    
    data = pd.read_csv(full_path)
    
    for category in data['Usage'].unique():
        print( "converting set: " + category + "...")
        # create folder
        if not os.path.exists(category):
            try:
                os.makedirs(OUTPUT_FOLDER_NAME + '/' + category)
            except OSError as e:
                if e.errno == errno.EEXIST and os.path.isdir(OUTPUT_FOLDER_NAME):
                   pass
                else:
                    raise
        
        # get samples and labels of the actual category
        category_data = data[data['Usage'] == category]
        samples = category_data['pixels'].values
        labels = category_data['emotion'].values
        
        # get images and extract features
        images = []
        for i in range(len(samples)):
            try:
                if labels[i] in SELECTED_LABELS: 
                    image = np.fromstring(samples[i], dtype=int, sep=" ").reshape((image_height, image_width))
                    images.append(image)
                    imageio.imwrite(OUTPUT_FOLDER_NAME + '/' + category + '/' + str(i) + '.jpg', image)
                    
            except Exception as e:
                print( "error in image: " + str(i) + " - " + str(e))
    
        np.save(OUTPUT_FOLDER_NAME + '/' + category + '/images.npy', images)

data_download()