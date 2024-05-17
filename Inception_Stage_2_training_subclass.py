# Step 1 : Git clone Feature map

# Clone from Github Repository
! git init .
! git remote add origin https://github.com/RichardMinsooGo/5_TF2_UCF101_video_classification.git
! git pull origin master
# ! git pull origin main

# Step 2 : Simple Classification using LSTM - Inception

import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.preprocessing import LabelBinarizer

import sys


BASE_PATH = './UCF101_Inception_64_frame'
# VIDEOS_PATH = os.path.join(BASE_PATH, '**','*.avi')
SEQUENCE_LENGTH = 64
        
######################################################
LABELS = ['UnevenBars','ApplyLipstick','TableTennisShot','Fencing','Mixing','SumoWrestling','HulaHoop','PommelHorse','HorseRiding','SkyDiving','BenchPress','GolfSwing','HeadMassage','FrontCrawl','Haircut','HandstandWalking','Skiing','PlayingDaf','PlayingSitar','FrisbeeCatch','CliffDiving','BoxingSpeedBag','Kayaking','Rafting','WritingOnBoard','VolleyballSpiking','Archery','MoppingFloor','JumpRope','Lunges','BasketballDunk','Surfing','SkateBoarding','FloorGymnastics','Billiards','CuttingInKitchen','BlowingCandles','PlayingCello','JugglingBalls','Drumming','ThrowDiscus','BaseballPitch','SoccerPenalty','Hammering','BodyWeightSquats','SoccerJuggling','CricketShot','BandMarching','PlayingPiano','BreastStroke','ApplyEyeMakeup','HighJump','IceDancing','HandstandPushups','RockClimbingIndoor','HammerThrow','WallPushups','RopeClimbing','Basketball','Shotput','Nunchucks','WalkingWithDog','PlayingFlute','PlayingDhol','PullUps','CricketBowling','BabyCrawling','Diving','TaiChi','YoYo','BlowDryHair','PushUps','ShavingBeard','Knitting','HorseRace','TrampolineJumping','Typing','Bowling','CleanAndJerk','MilitaryParade','FieldHockeyPenalty','PlayingViolin','Skijet','PizzaTossing','LongJump','PlayingTabla','PlayingGuitar','BrushingTeeth','PoleVault','Punch','ParallelBars','Biking','BalanceBeam','Swing','JavelinThrow','Rowing','StillRings','SalsaSpin','TennisSwing','JumpingJack','BoxingPunchingBag'] 
encoder = LabelBinarizer()
encoder.fit(LABELS)

######################################################

"""
model = tf.keras.Sequential([
    tf.keras.layers.Masking(mask_value=0.),
    tf.keras.layers.LSTM(512, dropout=0.5, recurrent_dropout=0.5),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(len(LABELS), activation='softmax')
])
"""

from tensorflow.keras.models import Model

class LSTM_Model(Model):
    def __init__(self):
        super(LSTM_Model, self).__init__()

        self.masking = tf.keras.layers.Masking(mask_value=0.)
        self.lstm    = tf.keras.layers.LSTM(512, dropout=0.5, recurrent_dropout=0.5)
        self.dense1  = tf.keras.layers.Dense(256, activation='relu')
        self.dropout = tf.keras.layers.Dropout(0.5)
        self.dense2  = tf.keras.layers.Dense(len(LABELS), activation='softmax')

    def call(self, x):
        print("Shape 1: ", x.shape)
        x = self.masking(x)
        print("Shape 2: ", x.shape)
        x = self.lstm(x)
        print("Shape 3: ", x.shape)
        x = self.dense1(x)
        print("Shape 4: ", x.shape)
        x = self.dropout(x)
        print("Shape 5: ", x.shape)
        out = self.dense2(x)
        print("Shape 6: ", x.shape)
        return out

model = LSTM_Model()


model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',             # optimizer='adam',
              metrics=['accuracy', 'top_k_categorical_accuracy'])


# model.summary()

model_name = 'UCF101_Video_Classification'

# import os.path
# if os.path.isfile(model_name+'.h5'):
#     model.load_weights(model_name+'.h5')

test_file = os.path.join('data', 'testlist01.txt')
train_file = os.path.join('data', 'trainlist01.txt')

with open('data/testlist01.txt') as f:
    test_list = [row.strip() for row in list(f)]

with open('data/trainlist01.txt') as f:
    train_list = [row.strip() for row in list(f)]
    train_list = [row.split(' ')[0] for row in train_list]

######################################################

def make_generator(file_list):
    def generator():
        np.random.shuffle(file_list)
        for path in file_list:
            full_path = os.path.join(BASE_PATH, path).replace('.avi', '.npy')
            # full_path = os.path.join(BASE_PATH, path)

            label = os.path.basename(os.path.dirname(path))
            features = np.load(full_path)

            padded_sequence = np.zeros((SEQUENCE_LENGTH, 2048))
            padded_sequence[0:len(features)] = np.array(features)

            transformed_label = encoder.transform([label])
            yield padded_sequence, transformed_label[0]
    return generator

train_dataset = tf.data.Dataset.from_generator(make_generator(train_list),
                 output_types=(tf.float32, tf.int16),
                 output_shapes=((SEQUENCE_LENGTH, 2048), (len(LABELS))))
train_dataset = train_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)


valid_dataset = tf.data.Dataset.from_generator(make_generator(test_list),
                 output_types=(tf.float32, tf.int16),
                 output_shapes=((SEQUENCE_LENGTH, 2048), (len(LABELS))))
valid_dataset = valid_dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir='./tmp', update_freq=1000)
    
model.fit(train_dataset, epochs=20, callbacks=[tensorboard_callback], validation_data=valid_dataset)

model.save_weights(model_name+'.h5', overwrite=True)

