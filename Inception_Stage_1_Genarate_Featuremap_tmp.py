
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

