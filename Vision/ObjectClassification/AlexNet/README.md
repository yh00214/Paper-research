# AlexNet 정리
## Abstract

- 1.2 million(1200만 개)의 고해상도 이미지 데이터셋인 ImageNet 데이터셋을 활용해, 1000개의 class를 분류하는 CNN 모델을 학습하였다.
- 당시 SOTA를 달성하였다.
- 5개의 CNN, 3개의 fully connected layer로 구성되어 있으며, 마지막에 softmax를 통해 1,000개의 class를 분류한다. CNN 사이엔 일부 max pooling이 존재한다. 총 6,000만 개 정도의 파라미터와 65만여 개의 뉴런이 존재한다.
- non-saturating neurons 기법과 efficient GPU implementation 기법을 적용해 학습이 빠르게 되도록 하였다.
- Overfitting을 방지하기 위해 Dropout을 적용하였다.

## 1. Introduction

- 최근 Object recognition 분야는 머신 러닝을 활용하는 게 필수적이게 됐다. 이를 위해 large dataset을 구하고 powerful model을 통해 학습을 하고 여러 기술을 통해 overfitting을 막아야 했으나, 지금까지는 large dataset을 구할 수 없었다. (CIFAR 등) 적은 dataset으로도 단순한 recognition은 잘 학습할 수 있었지만, 현실의 이미지들은 내부적으로 복잡한 특징들을 갖고 있어 학습하기 힘들었다. 하지만 최근 large dataset이 제공되기 시작하면서(ImageNet, LabelMe), 이런 이미지들도 학습 가능성이 열리기 시작했다.
- 수천만 장의 이미지를 학습하기 위해, 이들의 정보를 모두 담을 수 있는 large learning capacity model이 필요했다. 또한 ImageNet 데이터셋으로 학습해도 현실의 데이터는 이와 비교할 수 없을 정도로 수없이 많기 때문에, 이미지의 일반적인 특성(prior knowledge)도 잘 학습할 수 있어야 했다. 이를 위해 연구된 모델 중 CNN을 사용했는데, CNN은 모델 특성상 capacity를 확장시키기 좋고(Depth나 Breatdth를 확장), 자연의 이미지의 특징을 추측하는 데 용이했다(자연의 이미지는 변하지 않고, 한 픽셀은 주변 픽셀과 연관성이 있다). 또 FCN과 비교해 적은 파라미터와 connection을 갖고 있어 학습에 용이하다.
- 이런 점에도 불구하고 CPU로 CNN 연산을 진행하기엔 너무 비쌌다.(연산 시간이 너무 오래걸리고 비용이 많이 든다) 하지만 CNN 연산을 최적화 할 수 있는 최신 GPU의 발전으로 인해 large CNN 모델로 학습하는 것이 가능해졌다.
- 논문의 contribution으로 언급하는 내용은,
    1. 5개의 CNN과 3개의 FCN으로 구성된 largest convolutional neural network를 만들었다.(2012년임을 감안해 largest라고 부를 만 하다) 이를 통해 ImageNet dataset을 학습한 결과 SOTA를 달성하였다.
    2. highly-optimized GPU implementation을 사용했고 이외의 모든 연산은 일반적인 CNN 훈련 방법을 사용해 외부인들이 활용하기 편하게 하였다.
    3. 성능을 개선하고 학습 속도를 빠르게 하기 위한 여러 기법을 사용하였다.(Section 3에서 소개)
    4. large dataset을 씀에도 현재 모델의 크기가 overfitting을 유발할 수 있기 때문에, 특정 기술(Dropout)을 사용해 이를 해결하고자 하였다.
    
    이와 함께 현재 모델 구조에서 CNN layer를 1개만 제거해도(해당 layer의 파라미터가 전체의 1%도 안 됨에도 불구하고) 성능이 저하되는 것을 확인했다며, 현재 구조가 중요한 의미를 띠고 있다고 주장하였다. ( ※ 개인적인 생각으로, CNN layer 수가 5개면 이미지의 feature를 모두 추출하기엔 많은 숫자가 아니기에 layer를 줄였을 때 성능이 떨어진 게 아닐까 생각합니다. )
    
## 2. Dataset

- ILSVRC(ImageNet Large-Scale Visual Recognition Challenge)의 ImageNet 데이터셋을 사용했다.(120만 장의 train image, 5만 장의 validation image, 15만 장의 test image로 1,000개의 label을 분류해야 한다) 전체 ImageNet 데이터셋은 1,500만 장의 22,000개의 label이 존재하는데, 이 중 일부를 Challenge에 사용한 것이다.
- ImageNet 데이터셋은 해상도가 다양해서, 학습하기 전 256*256으로 downsampling 해줬다. 길이가 256보다 작을 경우 256이 되도록 rescale 시켜주고, 이후 가운데 영역을 256*256만큼 crop하였다.
- 모든 데이터에 대해 traing set의 pixel들의 mean 값을 각 pixel에서 빼주는 작업을 했고, 이 외 어떤 전처리도 진행하지 않았다. 즉 RGB 값은 유지한 채 학습을 진행했다.

## 3. Architecture

![Untitled](https://user-images.githubusercontent.com/33994833/169072195-27d4f3f4-c004-4c38-9462-9f5e60d67e70.png)

- AlexNet에 사용된 중요한 기법들
    - ReLU Activation(논문에선 Nonlinearity란 표현을 사용했지만 편의상 Activation function이란 표현을 사용하겠습니다.) : 기존에 많이 사용되던 Activation function인 tanh 와 sigmoid대신, ReLU를 사용했을 때 몇 배 더 빠른 학습 속도를 기록할 수 있었다.
![Untitled](https://user-images.githubusercontent.com/33994833/169072458-c141da72-4271-49d6-a61b-1b521c564c9c.png)



