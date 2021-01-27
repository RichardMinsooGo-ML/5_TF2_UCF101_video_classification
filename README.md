# KR>

해당 저장소는 UCF101 Video 데이터들을 모아 놨습니다.

## Option 1 

필자가 운영중인 구글 드라이브를 통해서 다운로드가 가능하다.

https://drive.google.com/drive/folders/1HLa2cqeX-vMMy5yI9S5B1aTgVX-ZcU0c?usp=sharing

UCF101.zip                      : Source video files

UCF101_MobilenetV2_64_frame.zip : feature vector generated with Mobilenet

UCF101_Inception_64_frame.zip   : feature vector generated with Inception

UCF101_Resnet152_64_frame.zip   : feature vector generated with Resnet

위의 네개의 파일을 다운로드 받은 후에 각각의 폴더에 압축을 풀어 준다.

## Option 2

해당 저장소를 다운로드 하면 원본 비디오와 Feature vector 파일을 얻을수 있습니다.


## 사용 방법

학습의 편의를 위하여 파이썬 스크립트는 두부분으로 되어 있습니다.


### Step 1 : Generation of feature vector / option 1

특징벡터 파일은 이미 세개의 폴더에 (UCF101_Inception_64_frame, UCF101_MobilenetV2_64_frame, UCF101_Resnet152_64_frame)생성 시켜 놓았습니다.

### Step 1 : Generation of feature vector / option 2

Feature vector는 파이썬 파일을 실행시켜서 얻을수는 있으나 시간이 많이 걸린다. GPU 가 없으면 시도를 안하는것이 낫다.

    python Create_feature_vector.py

위와 같이 실행시키면 default로 MobilenetV2 로 만들어진 feature vector를 얻을수 있다.

Vector_name = 'Inception' 을 선택하면, Inception으로 생성된 feature vector,

Vector_name = 'ResNet152' 을 선택하면, ResNet152으로 생성된 feature vector를 얻을수 있다.

### Step 2 : Training from feature vecter

    python Train_from_feature_vector.py

를 실행시켜 학습이 진행되는것을 살펴 보자. 비디오 파일의 특성상 LSTM 을 적용하여 네트워크를 구성하였다.

Default로 MobilenetV2 로 만들어진 feature vector를 학습하도록 설정 하였다.

Vector_name = 'Inception' 을 선택하면, Inception으로 생성된 feature vector,

Vector_name = 'ResNet152' 을 선택하면, ResNet152으로 생성된 feature vector를 사용하여 학습하도록 할수 있다.


# EN> 

This repository wil be used as UCF101 datasets.

## Option 1 

You can download at my Google Drive. Visit below link and download

https://drive.google.com/drive/folders/1HLa2cqeX-vMMy5yI9S5B1aTgVX-ZcU0c?usp=sharing

UCF101.zip                      : Source video files

UCF101_MobilenetV2_64_frame.zip : feature vector generated with Mobilenet

UCF101_Inception_64_frame.zip   : feature vector generated with Inception

UCF101_Resnet152_64_frame.zip   : feature vector generated with Resnet

After the dwonload, please unzip at each folder.


## Option 2

Also, you can directly download this repository for source video and vecture vectors


## How to use

To speed up the training, python scrpts are divided with 2 parts.

### Step 1 : Generation of feature vector / option 1

Already I creted the feature vector array at three folders (UCF101_Inception_64_frame, UCF101_MobilenetV2_64_frame, UCF101_Resnet152_64_frame).

### Step 1 : Generation of feature vector / option 2

You can get feature vectors by using python files like below. But it takes very long time. Only try if you have GPU.

    python Create_feature_vector.py

As default, feature vectors are genetated by MobilenetV2.

Vector_name = 'Inception' generate feature vectors by Inception.

Vector_name = 'ResNet152' generate feature vectors by ResNet152.

### Step 2 : Training from feature vecter

You can test the training from feature vector.

    python Train_from_feature_vector.py

Due to the charateristics of video, model was designed with LSTM.

As default, it will train with feature vectors from MobilenetV2.

Vector_name = 'Inception' will train with feature vectors from by Inception.

Vector_name = 'ResNet152' will train with feature vectors from by ResNet152.


# Other Repositories

https://github.com/RichardMinsooGo
: This repository will be used as basics of TF2

https://github.com/RichardMinsooGo-ML
: This is new repository for Machine Learning.


https://github.com/RichardMinsooGo-RL-Gym
: This is new repository for Reinforcement Learning based on Open-AI gym.


https://github.com/RichardMinsooGo-RL-Single-agent  
: This is new repository for Reinforcement Learning for Single Agent.


https://github.com/RichardMinsooGo-RL-Multi-agent
: This new repository is for Reinforcement Learning for Multi Agents.

