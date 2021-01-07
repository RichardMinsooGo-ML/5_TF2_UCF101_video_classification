## KR>

해당 저장소는 UCF101 Video 데이터들을 모아 놨습니다.

학습의 편의를 위하여 파이썬 스크립트는 두부분으로 되어 있습니다.

특징벡터 파일은 이미 세개의 폴더에 (UCF101_Inception_64_frame, UCF101_MobilenetV2_64_frame, UCF101_Resnet152_64_frame)생성 시켜 놓았습니다.

또한 1_create_feature_vector_xxxxx.py를 실행시켜서 얻는 방법이 있지만, 이는 시간이 너무 오래 걸리므로 추천하지는 않습니다.
이 파이썬 스크립트는 고성능 네트워크를 구성하기 위해서 만들어 놓은 부분입니다.

성능이 낮은 컴퓨터를 사용하시는 개발자 분들은 "2_train_test_xxxxx.py"만 사용하여 학습을 진행하시면 됩니다.

Video 이 특성상, LSTM 을 사용하여 구성하였습니다.


## EN> 

This repository wil be used as UCF101 datasets.

To speed up the training, python scrpts are divided with 2 parts.

Already I creted the feature vector array at three folders (UCF101_Inception_64_frame, UCF101_MobilenetV2_64_frame, UCF101_Resnet152_64_frame).

Also, you can generate the feature map with 1_create_feature_vector_xxxxx.py. But it takes too long time even with sigle GPU.

If you have just single GPU, please only try with "2_train_test_xxxxx.py".

Due to the chracterisic of Video, training will be done with LSTM.


## Other Repository
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

