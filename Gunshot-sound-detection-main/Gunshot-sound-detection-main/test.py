import sys
import os
import string
import random
import operator
from werkzeug.utils import secure_filename
import pandas as pd
import numpy as np
import tensorflow as tf
import librosa
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, MaxPooling2D, GlobalAveragePooling2D
import soundfile as sf

from sos import alert


featuresdf = pd.read_hdf('./models/dataframes_backup.h5')

classes = {'1': 'gun_shot', '0': 'no_gun_shot'}
################################### Extract features ###########################
def extract_features(file_name):

    try:
        # audio, sample_rate = librosa.load(file_name, res_type='kaiser_fast')
        audio, sample_rate = sf.read(file_name)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        print(mfccs)
        mfccsscaled = np.mean(mfccs.T,axis=0)
        
    except Exception as e:
        print(e)
        return None

    return mfccsscaled

def extract_features_arr(audio_arr, sample_rate):

    mfccs = librosa.feature.mfcc(y=audio_arr, sr=sample_rate, n_mfcc=40)
    mfccsscaled = np.mean(mfccs.T,axis=0)

    return mfccsscaled
################################################################################

import tensorflow as tf
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

num_rows = 4
num_columns = 10
num_channels = 1
num_labels = 2

# Construct model
model = Sequential()
model.add(Conv2D(filters=16, kernel_size=2, padding="same", input_shape=(num_rows, num_columns, num_channels), activation='relu'))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))

model.add(Conv2D(filters=32, kernel_size=2, padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))

model.add(Conv2D(filters=64, kernel_size=2, padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))

model.add(Conv2D(filters=128, kernel_size=2, padding="same", activation='relu'))
model.add(MaxPooling2D(pool_size=1))
model.add(Dropout(0.2))
model.add(GlobalAveragePooling2D())

model.add(Dense(num_labels, activation='sigmoid'))

model.load_weights("./models/best_weights_temp.hdf5")

def predict_class(featuresdf):
    # a=model.predict(featuresdf.reshape(1, 4, 10, 1))
    # print(a)
    pred_index_tensor=tf.argmax(model.predict(featuresdf.reshape(1, 4, 10, 1)), axis=1)
    pred_index_arr = pred_index_tensor.numpy()
    pred_index = str(pred_index_arr[0])
    pred_class = classes[pred_index]
    return (pred_index, pred_class)


# file_path='gun3.wav'

# f_features = extract_features(file_path)
# print(f_features)
# class_index, class_name = predict_class(f_features)
# print("\tPredicted Class: " + str(class_index))
# print("\tPredicted Class Name: " + str(class_name))



##########################################


import queue 
prediction_history = queue.Queue(maxsize=5) 

sample_rate = 44100
threshold = 1 # required number of gunshots, in most recent 5 predictions, 
              # to make final decision
while True:
    gunshot_count = 0
    from rcaudio import CoreRecorder
    CR = CoreRecorder(
            time = 4, # How much time to record
            sr = sample_rate # sample rate
            )
    CR.start()
    # CR.stop()
    
    data = [0] * 44100 # initializing the overlapping sound with zeros
    
    while True:
        if not CR.buffer.empty():
            # get every integer from buffer and append in data list
            x = CR.buffer.get()
            data.append(x)
            
            # 4 seconds of recorded sound and 1 second of overlapping sound
            if len(data)//44100 == 5:
                break
                
    data = np.array(data, dtype="float32")

    arr_features = extract_features_arr(data, sample_rate)
    class_index, class_name = predict_class(arr_features)
    print(class_index, class_name)
    if(class_index==1):
        break
    data = data.tolist()[-44100:] 
    
    prediction_history.put(class_index)
    
    if prediction_history.full():
        while prediction_history.empty() != True:
            elem = prediction_history.get()
            if elem == "1": # gunshot detected
                gunshot_count += 1
        
        if gunshot_count >= threshold:
            print("Gunshots Detected!")
            # alert()##################################################################
        else:
            print('peaceful')
