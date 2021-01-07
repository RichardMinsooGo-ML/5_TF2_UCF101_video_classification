import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import cv2
import numpy as np
import time
# from PIL import Image
import shutil
from shutil import copytree, ignore_patterns

import sys

image_size = 299
n_feature_frame = 64
dim = (image_size, image_size)

src_name = 'UCF101'
dst_name = 'UCF101_MobilenetV2_64_frame'

src_folder = os.path.join("./", src_name) 
dst_folder = os.path.join("./", dst_name)

if os.path.exists(dst_folder):
    shutil.rmtree(dst_folder)
    
# sys.exit()
    
# copytree(src_folder, dst_folder, ignore=ignore_patterns('*.avi', 'tmp*'))
copytree(src_folder, dst_folder, ignore=ignore_patterns('*.*'))

total_file_count = sum(len(files) for _, _, files in os.walk(src_folder))

# print(total_file_count)

# sys.exit()

base_model = tf.keras.applications.MobileNetV2(include_top=False, weights='imagenet')

x = base_model.output

# We add Average Pooling to transform the feature map from
# 8 * 8 * 2048 to 1 x 2048, as we don't need spatial information
pooling_output = tf.keras.layers.GlobalAveragePooling2D()(x)

feature_extraction_model = tf.keras.Model(base_model.input, pooling_output)

start_time = time.time()

# print(start_time)
# sys.exit()
for root, dirs, files in os.walk(src_folder):
    file_count = 0
    for file in files:
        file_count += 1
        
        with open(os.path.join(root, file), "r") as auto:
            
            # print(file)
            # sys.exit()
            
            # 영상의 의미지를 연속적으로 캡쳐할 수 있게 하는 class
            
            vidcap = cv2.VideoCapture(file)
            count = 0
            save_count = 0
            datasets =np.zeros((n_feature_frame, image_size, image_size,3)) 

            # cap = cv2.VideoCapture("video.mp4")
            video_len = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))
            # print( video_len )
            while count < video_len:

                ret, image = vidcap.read()
                frame_count = int(count / (video_len/n_feature_frame))
                if save_count == frame_count:

                    # OPENCV reads in BGR, tensorflow expects RGB so we invert the order

                    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
                    # resize = cv2.resize(image, (160, 120))
                    # cv2.imwrite("./images/frame%d.jpg" % frame_count, image)

                    datasets[save_count] = image

                    # print('Saved frame%d.jpg' % frame_count,"at count", count)
                    save_count += 1


                    # print(image.shape)
                    # print(datasets.shape)

                    # sys.exit()

                else:
                    pass
                count += 1

            datasets = datasets/255.   

            batch_features = feature_extraction_model(datasets)
            batch_features = np.array(batch_features)
            
            # print(batch_features[:,0:6])
            # print(batch_features[:,2042:2048])

            # vidcap.release()

            root_dst = root.replace(src_name, dst_name)
            file_dst = file.replace("avi", "npy")
            
            np.save(root_dst+'/'+file_dst, batch_features)
            
            # root_dst = str(root_dst.as_posix())
            root_dst = root_dst.replace(os.sep, '/')
            # root_dst = root_dst.replace('\\', '/')
            current_time = time.time()
            left_time = int(((current_time - start_time)/file_count)*(total_file_count- file_count))
            
            # e = int(time.time() - start_time)
            # print(' Elasped time :{:02d}:{:02d}:{:02d}'.format(e // 3600, (e % 3600 // 60), e % 60))            
            
            print("Converted:{:>6,d}".format(file_count),"/","{:>6,d}".format(total_file_count),
                  '/ Expected remain time:{:02d}:{:02d}:{:02d}'.format(left_time // 3600, (left_time % 3600 // 60), left_time % 60),
                  "/ Location:",root_dst+'/'+file_dst)
            
            # if file_count == 10:
            #     sys.exit()
            
            
            # file = np.load('./v_BalanceBeam_g01_c01.npy')
            # print(file.shape)















