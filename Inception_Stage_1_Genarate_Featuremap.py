# Step1. Download and extract UCF101 Data

! wget --no-check-certificate https://www.crcv.ucf.edu/data/UCF101/UCF101.rar
% rm -rf sample_data
! mkdir UCF101
! unrar x "/content/UCF101.rar" "/content/UCF101"

# Screen Cleaning, there are too many files to display.
from IPython.display import clear_output 
clear_output()

# Step2. [Option] Delete unused video files

% rm -rf /content/UCF101/UCF-101/YoYo
% rm -rf /content/UCF101/UCF-101/WritingOnBoard
% rm -rf /content/UCF101/UCF-101/WallPushups
% rm -rf /content/UCF101/UCF-101/WalkingWithDog
% rm -rf /content/UCF101/UCF-101/VolleyballSpiking
% rm -rf /content/UCF101/UCF-101/UnevenBars
% rm -rf /content/UCF101/UCF-101/Typing
% rm -rf /content/UCF101/UCF-101/TrampolineJumping
% rm -rf /content/UCF101/UCF-101/ThrowDiscus
% rm -rf /content/UCF101/UCF-101/TennisSwing
% rm -rf /content/UCF101/UCF-101/TaiChi
% rm -rf /content/UCF101/UCF-101/TableTennisShot
% rm -rf /content/UCF101/UCF-101/Swing
% rm -rf /content/UCF101/UCF-101/Surfing
% rm -rf /content/UCF101/UCF-101/SumoWrestling
% rm -rf /content/UCF101/UCF-101/StillRings
% rm -rf /content/UCF101/UCF-101/SoccerPenalty
% rm -rf /content/UCF101/UCF-101/SoccerJuggling
% rm -rf /content/UCF101/UCF-101/SkyDiving
% rm -rf /content/UCF101/UCF-101/Skijet
% rm -rf /content/UCF101/UCF-101/Skiing
% rm -rf /content/UCF101/UCF-101/SkateBoarding
% rm -rf /content/UCF101/UCF-101/Shotput
% rm -rf /content/UCF101/UCF-101/ShavingBeard
% rm -rf /content/UCF101/UCF-101/SalsaSpin
% rm -rf /content/UCF101/UCF-101/Rowing
% rm -rf /content/UCF101/UCF-101/RopeClimbing
% rm -rf /content/UCF101/UCF-101/RockClimbingIndoor
% rm -rf /content/UCF101/UCF-101/Rafting
% rm -rf /content/UCF101/UCF-101/PushUps
% rm -rf /content/UCF101/UCF-101/Punch
% rm -rf /content/UCF101/UCF-101/PullUps
% rm -rf /content/UCF101/UCF-101/PommelHorse
% rm -rf /content/UCF101/UCF-101/PoleVault
% rm -rf /content/UCF101/UCF-101/PlayingViolin
% rm -rf /content/UCF101/UCF-101/PlayingTabla
% rm -rf /content/UCF101/UCF-101/PlayingSitar
% rm -rf /content/UCF101/UCF-101/PlayingPiano
% rm -rf /content/UCF101/UCF-101/PlayingGuitar
% rm -rf /content/UCF101/UCF-101/PlayingFlute
% rm -rf /content/UCF101/UCF-101/PlayingDhol
% rm -rf /content/UCF101/UCF-101/PlayingDaf
% rm -rf /content/UCF101/UCF-101/PlayingCello
% rm -rf /content/UCF101/UCF-101/PizzaTossing
% rm -rf /content/UCF101/UCF-101/ParallelBars
% rm -rf /content/UCF101/UCF-101/Nunchucks
% rm -rf /content/UCF101/UCF-101/MoppingFloor
% rm -rf /content/UCF101/UCF-101/Mixing
% rm -rf /content/UCF101/UCF-101/MilitaryParade
% rm -rf /content/UCF101/UCF-101/Lunges
% rm -rf /content/UCF101/UCF-101/LongJump
% rm -rf /content/UCF101/UCF-101/Knitting
% rm -rf /content/UCF101/UCF-101/Kayaking
% rm -rf /content/UCF101/UCF-101/JumpingJack
% rm -rf /content/UCF101/UCF-101/JumpRope
% rm -rf /content/UCF101/UCF-101/JugglingBalls
% rm -rf /content/UCF101/UCF-101/JavelinThrow
% rm -rf /content/UCF101/UCF-101/IceDancing
% rm -rf /content/UCF101/UCF-101/HulaHoop
% rm -rf /content/UCF101/UCF-101/HorseRiding
% rm -rf /content/UCF101/UCF-101/HorseRace
% rm -rf /content/UCF101/UCF-101/HighJump
% rm -rf /content/UCF101/UCF-101/HeadMassage
% rm -rf /content/UCF101/UCF-101/HandstandWalking
% rm -rf /content/UCF101/UCF-101/HandstandPushups
% rm -rf /content/UCF101/UCF-101/Hammering
% rm -rf /content/UCF101/UCF-101/HammerThrow
% rm -rf /content/UCF101/UCF-101/Haircut
% rm -rf /content/UCF101/UCF-101/GolfSwing
% rm -rf /content/UCF101/UCF-101/FrontCrawl
% rm -rf /content/UCF101/UCF-101/FrisbeeCatch
% rm -rf /content/UCF101/UCF-101/CliffDiving
% rm -rf /content/UCF101/UCF-101/CricketBowling
% rm -rf /content/UCF101/UCF-101/CricketShot
% rm -rf /content/UCF101/UCF-101/CuttingInKitchen
% rm -rf /content/UCF101/UCF-101/Diving
% rm -rf /content/UCF101/UCF-101/Drumming
% rm -rf /content/UCF101/UCF-101/Fencing
% rm -rf /content/UCF101/UCF-101/FieldHockeyPenalty
% rm -rf /content/UCF101/UCF-101/FloorGymnastics
% rm -rf /content/UCF101/UCF-101/Billiards
% rm -rf /content/UCF101/UCF-101/BlowDryHair
% rm -rf /content/UCF101/UCF-101/BlowingCandles
% rm -rf /content/UCF101/UCF-101/BodyWeightSquats
% rm -rf /content/UCF101/UCF-101/Bowling
% rm -rf /content/UCF101/UCF-101/BoxingPunchingBag
% rm -rf /content/UCF101/UCF-101/BoxingSpeedBag
% rm -rf /content/UCF101/UCF-101/BreastStroke
% rm -rf /content/UCF101/UCF-101/BrushingTeeth
% rm -rf /content/UCF101/UCF-101/CleanAndJerk

# Step3. Build mirror directories

import os

inputpath = '/content/UCF101/UCF-101/'
outputpath_1 = './UCF101_Inception_64_frame/'

for dirpath, dirnames, filenames in os.walk(inputpath):
    structure = os.path.join(outputpath_1, dirpath[len(inputpath):])
    if not os.path.isdir(structure):
        os.mkdir(structure)
    else:
        print("Folder does already exits!")

outputpath_2 = './UCF101_MobilenetV2_64_frame/'

for dirpath, dirnames, filenames in os.walk(inputpath):
    structure = os.path.join(outputpath_2, dirpath[len(inputpath):])
    if not os.path.isdir(structure):
        os.mkdir(structure)
    else:
        print("Folder does already exits!")

outputpath_3 = './UCF101_Resnet152_64_frame/'

for dirpath, dirnames, filenames in os.walk(inputpath):
    structure = os.path.join(outputpath_3, dirpath[len(inputpath):])
    if not os.path.isdir(structure):
        os.mkdir(structure)
    else:
        print("Folder does already exits!")

# Step4. Generate feature map - Inception

import tensorflow as tf
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import tqdm
from sklearn.preprocessing import LabelBinarizer

import sys


BASE_PATH = './UCF101_Inception_64_frame'
VIDEOS_PATH = os.path.join('/content/UCF101/UCF-101', '**','*.avi')
SEQUENCE_LENGTH = 64

def frame_generator():
    video_paths = tf.io.gfile.glob(VIDEOS_PATH)
    np.random.shuffle(video_paths)
    for video_path in video_paths:
        frames = []
        cap = cv2.VideoCapture(video_path)
        num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        sample_every_frame = max(1, num_frames // SEQUENCE_LENGTH)
        current_frame = 0

        label = os.path.basename(os.path.dirname(video_path))

        max_images = SEQUENCE_LENGTH
        while True:
            success, frame = cap.read()
            if not success:
                break

            if current_frame % sample_every_frame == 0:
                # OPENCV reads in BGR, tensorflow expects RGB so we invert the order
                frame = frame[:, :, ::-1]
                img = tf.image.resize(frame, (299, 299))
                img = tf.keras.applications.inception_v3.preprocess_input(
                    img)
                max_images -= 1
                yield img, video_path

            if max_images == 0:
                break
            current_frame += 1

# `from_generator` might throw a warning, expected to disappear in upcoming versions:
# https://www.tensorflow.org/versions/r2.0/api_docs/python/tf/data/Dataset#for_example_2
dataset = tf.data.Dataset.from_generator(frame_generator,
             output_types=(tf.float32, tf.string),
             output_shapes=((299, 299, 3), ()))

dataset = dataset.batch(16).prefetch(tf.data.experimental.AUTOTUNE)

inception_v3 = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')

x = inception_v3.output

# We add Average Pooling to transform the feature map from
# 8 * 8 * 2048 to 1 x 2048, as we don't need spatial information
pooling_output = tf.keras.layers.GlobalAveragePooling2D()(x)

feature_extraction_model = tf.keras.Model(inception_v3.input, pooling_output)

current_path = None
all_features = []

for img, batch_paths in tqdm.tqdm(dataset):
    batch_features = feature_extraction_model(img)
    batch_features = tf.reshape(batch_features, 
                              (batch_features.shape[0], -1))
    
    for features, path in zip(batch_features.numpy(), batch_paths.numpy()):
        if path != current_path and current_path is not None:
            
            # print("\nOOO PATH ",current_path)
            # print("CUR PATH :", current_path.decode())
            
            output_path = current_path.decode().replace('.avi', '.npy')
            output_path = output_path.replace('UCF101/UCF-101', 'UCF101_Inception_64_frame')

            # print("OUT PAT :", output_path)

            np.save(output_path, all_features)
            all_features = []
            
            # import sys
            # sys.exit()


        current_path = path
        all_features.append(features)

