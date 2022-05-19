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
    - ReLU Activation(논문에선 Nonlinearity란 표현을 사용했지만 편의상 Activation function이란 표현을 사용하겠습니다.)
        - 기존에 많이 사용되던 Activation function인 tanh 와 sigmoid대신, ReLU를 사용했을 때 몇 배 더 빠른 학습 속도를 기록할 수 있었다.
        - 논문에서 tanh와 sigmoid를 saturating nonlinearity라 표현하고, ReLU를 non-saturating nonlinearity라 표현하는데, saturating은 x가 무한으로 갈 때 y가 일정 값에 수렴하는 형태를 의미하고, non-saturating은 x가 무한으로 갈 때 y도 무한히 발산하는 경우를 의미한다.
        - 아래의 이미지에서 Training Error가 25%까지 떨어지는 데에 ReLU(실선)이 tanh(점선)에 비해 6배 이상 빠른 것을 볼 수 있다. 저자는 만약 전통적인 activation function(tanh나 sigmoid)를 썼으면 해당 모델을 실험하기 힘들었을 것이라고 말했다.
![Untitled](https://user-images.githubusercontent.com/33994833/169072458-c141da72-4271-49d6-a61b-1b521c564c9c.png)

    - 여러 GPU를 활용한 훈련
        - 논문에서 사용한 GPU인 GTX 580은 메모리가 3GB 밖에 되지 않아서, 모든 네트워크 구조를 담을 수 없었다. 따라서 네트워크를 둘로 나눠 2개의 GPU에 나눠 담아 연산하였다. GPU를 이용하면 호스트 메모리 접근 없이 GPU간 메모리에 직접 접근하여 읽고 쓸 수 있었기 때문에 cross-GPU parallelization 방법을 사용하기 용이했다.
        - 두 개의 GPU에 kernel을 절반씩 나눠 배치한 후, GPU 간에는 특정 계층에서만 통신하도록 하였다. Architecture 그림을 보면, Layer 2 → 3으로 갈 때와 FCN 단계에서 GPU간 통신이 이루어지는 것을 볼 수 있다.
        - 단일 GPU에서 kernel을 절반만 사용해 학습하는 경우와 비교했을 때(kernel 전체를 사용하는 건 메모리 한계때문에 애초에 불가했으니 이렇게 비교한 것 같다.), GPU 2개를 활용해 parallelization을 구현한 현 모델이 Top-5, Top-1 error rate가 1.2%, 1.7% 낮았다. 학습 시간도 GPU 2개 쪽이 조금 더 빨랐다.
    - Local Response Normalization
        - ReLU는 non-saturating하기 때문에 Input normalization 없이도 사용 가능하다. saturating하면 일정 크기 이상의 값들은 전부 비슷한 크기로 변환이 될 것이기 때문에 반드시 이를 해결해줘야 하지만, ReLU는 Input으로 양수 값이 들어오면 값을 그대로 유지하기 때문이다. 하지만 Local Response Normalization을 적용함으로서 일반화에 좀 더 도움이 됨을 알 수 있었다.
        - (논문에 없는 내용) 특정 위치의 ReLU input 값이 크고, 주변 값들이 작다면, 큰 값이 주변의 다른 값에 영향을 끼칠 수 있기 때문에, 이를 방지하기 위해 주변에 있는 pixel들 간에 정규화를 해준다고 한다.
        - normalization이지만 각 위치마다 그룹의 평균을 빼는 작업이 있진 않아 “bright normalization”이라 주장하며, 이를 사용하였을 때 top-1과 top5 error rate에서 각각 1.4%, 1.2%의 성능 향상이 있었다고 한다.
        - ※ 현재는 잘 사용되지 않고, Batch Normalization으로 대체된 것으로 알고있다.
        
        ![Untitled](https://user-images.githubusercontent.com/33994833/169382662-7093724f-8452-4369-9a42-6a899c05d16a.png)
    - Overlapping Pooling
        - 일반적인 pooling 기법들은 인접한 pooling간에 겹치지 않게(not overlap) 진행이 된다. 즉, stride 크기 s와 filter size z가 있을 때, 일반적으론 s=z 인 경우가 많다. 하지만 이 논문에선 s를 z보다 작게 하였다. s=2, z=3으로 실험을 진행했고 이는 3x3 필터를 stride 2만큼 움직여가며 pooling을 진행하는 것으로, pooling 간에 겹치는 영역이 조금씩 생기게 되었다. 그래서 이를 Overlapping pooling이라 부른다. 이를 통해 s=2, z=2인 경우와 비교해  top-1과 top-5 error rate가 0.4%, 0.3% 줄어들었고, 실험동안 Overlapping pooling을 사용한 모델이 다른 경우에 비해 overfitting 되는 경우가 좀 더 줄어들었다고 주장하였다.
- 전체 Architecture를 이 그림과 함께 다시 설명해보자.
![Untitled](https://user-images.githubusercontent.com/33994833/169383017-f4d89b21-aba8-4a07-913d-40d21b2f1e17.png)
    - 5개의 Convolution layer와 3개의 Fully connected layer로 구성되어 있다.
    - 마지막 FCN을 통과한 후 나온 1000개의 값에 Softmax를 취해줌으로서 1000개의 class에 대한 확률 값을 얻을 수 있다.
    - 2, 4, 5번째 Conv layer의 kernel은 이전 layer 중 같은  GPU에 있는 kernel에만 연결되어 있다.
    - 3번째 Conv layer의 kernel과 모든 FCN은 이전 layer의 각 GPU에 있는 kernel과 모두 연결되어 있다.
    - LRN(Local Response Normalization)은 1, 2번째 Conv layer 이후에 적용된다.
    - Max-pooling(Overlapping pooling)은 모든 LRN 이후와 5번째 Conv layer 이후에 적용된다.
    - ReLU non-linearity는 모든 layer의 output에 적용된다.
        - 순서 상 Layer → ReLU → LRN → Max pooling 순으로 적용된다.
    - 그림에서 이전 layer에 표시된 직육면체 또는 kernel이 현재 layer의 kernel을 의미한다. 즉 1번째 Conv layer는 11*11*3 (stride 4)의 kernel을 96개(두 GPU 통합) 가지고, 2번째 layer는 5*5*48(한 GPU의 kernel만 사용하기 때문)의 kernel로 256개의 channel을 만든다. 3번째 layer는 두 GPU 모두로부터 정보를 가져오므로 kernel size는 3*3*256이 된다. 이 후 layer에도 동일하게 적용하면 된다.
    - FCN layer는 각 4096개의 neuron을 갖고 있다.

## 4. Reduce Overfitting

- 모델에 사용된 6,000만 개의 파라미터를 현 데이터셋 그대로 학습하면서 Overfitting이 없는 건 어려운 일이었다.(120만 장의 train data를 활용해도, 가능한 모든 이미지의 general feature를 학습하는 건 어렵고, 파라미터 수가 많은 편이기 때문에 train data에 맞는 특징만 학습할 수도 있다. 당시엔 파라미터 수가 많으면 쉽게 Overfitting이 일어났기에 이런 생각을 한 듯 하다.) 아래에 소개된 기법을 통해 Overfitting을 막으려 노력하였다.
    - Data Augmentation
        - 두 가지 형태의 Augmentation을 적용하였으며, 이들은 원래 이미지를 이용해 아주 적은 computation으로 구할 수 있어 따로 디스크에 변환된 이미지를 저장할 필요가 없다. 또, Augmentation은 CPU에서 실행하여, GPU가 이전 batch로 학습을 진행하는 와중에 CPU에서 Python 코드를 실행해 다음 batch의 이미지들을 만들게 설계하여, GPU 연산 할당으로부터 자유롭다.
        - 첫 번째 Augmentation 방법은 crop된 이미지 생성 및 좌우 뒤집기(horizontal reflection) 한 것이다. Input 이미지의 크기는 256*256인데, 이미지 내에서 random하게 224*224만큼 crop하여 crop한 이미지와 좌우로 뒤집힌 crop 이미지 두 개의 patch를 생성할 수 있다. 만들어진 이미지들은 비슷하긴 하지만(inter-dependent), 이를 통해 training dataset을 2048배 증가시켜 Overfitting 하는 것을 막았다. 테스트 단계에서는 5개의 224*224 patch를  4개의 코너와 center로부터 추출하고, 반사된 이미지까지 총 10개의 patch에 대해 model 통과 후 만들어진 softmax 예측을 평균 내서 원 이미지의 class를 예측하였다.
        - 데이터의 RGB 값에 PCA를 적용하여, 각 주성분의 크기에 비례하게 평균 0, 표준 편차 0.1인 eigenvalue를 곱해주었다. 이를 통해 이미지의 조명 또는 색깔이 변화하지만, 이는 약간의 조명과 색깔 변화로도 불변하는 이미지의 중요한 property를 찾는 데 도움을 준다고 주장하였다. 이를 활용해 top-1 error를 1% 이상 감소시킨다.
    - Dropout
        - 서로 다른 모델의 예측 결과를 종합하여 최종 예측을 하면 test error를 줄일 수 있다. 하지만 하나의 네트워크를 학습하는 데도 며칠이 걸리는데(논문 기준), 여러 개의 모델을 학습하는 것은 시간적으로 낭비이다. 하지만 Dropout이란 기법을 사용하면 2배의 시간만을 사용하고도 모델들을 Combination한 효과를 얻을 수 있다.
        - Dropout은 모델의 hidden neuron의 출력 확률을 0.5의 확률로 0으로 만드는 방법이다. 0이 된 뉴런은 forward와 backward pass 모두 진행되지 않는다.(말 그대로 없는 뉴런 취급을 당한다)
        - Input이 들어올 때마다 Dropout을 통해 다른 모델 architecture가 만들어지지만, 출력이 0이 되든 아니든 neuron의 weight 값은 공유한다.
        - 모델 구조가 항상 똑같으면, neuron들은 최종 output을 예측하는 데 있어 서로 의존적이게 될 수 있다.(A라는 결과를 얻기 위해 뉴런 a,b,c,d,e 모두의 output이 조금씩 필요한 상황) 하지만 Dropout을 통해 모델 구조가 매번 바뀌면, neuron간의 의존성이 줄고 neuron 스스로가 좀 더 input의 robust한 특성을 학습하려고 하게 된다.(A라는 결과를 얻기 위해, a,b만 보고, 혹은 a만 보고 어느정도 예측할 수 있게 된다고 할 수 있을 것 같다. 말이 찰떡 같진 않지만, 적은 neuron 수로도 output을 잘 예측하기 위해 각 neuron이 좀 더 robust한 특징을 보게 된다고 생각하면 될 것 같다.)
        - Test 시에는 모든 neuron을 prediction에 사용하고, 대신 output들에 0.5를 곱한다. 원래 절반의 neuron들로도 예측을 잘 하도록 학습하였기에, 2배의 neuron을 사용하게 됐으니 output들이 좀 더 train data로 학습했을 떄의 분포를 따라가도록 한 것 같다.(논문에서는 학습 과정에서 기하급수적으로 많이 생긴 Dropout Network로부터 생긴 예측 분포의 기하학적 평균에 대한 합리적인 근사치가 output들에 0.5를 곱하는 것이라고 한다.)
        - AlexNet은 첫 두개의 FCN에 Dropout을 적용하였다. Dropout을 사용하면 학습이 수렴하기까지 2배 정도 반복횟수가 더 필요했다. 하지만 Dropout을 사용하지 않으면 모델은 상당히 Overfitting 되었다.

## 5. Details of learning
![Untitled](https://user-images.githubusercontent.com/33994833/169384090-7b6404b4-8346-4b70-a11b-b98a6b60450f.png)
- AlexNet의 Optimizer로 SGD(Stochastic Gradient Descent)를 사용하였다. batch size=128, momentum=0.9, weight decay=0.0005로 설정하였다. 이 때, 저자는 Weight decay가 학습에 중요한 영향을 끼친다는 것을 발견했는데, 단지 regularizer일 뿐만 아니라 모델의 training error를 줄이는 효과를 가져왔다.  $i$는 iteration index(epoch), $v$는 momentum value, $\epsilon$은 learning rate, $\left<\frac{\partial L}{\partial w}|_{w_i}\right>_{D_i}$는 i번째 batch D에서, $w_i$시점에서의 w에 대한 미분값의 평균을 의미한다.
- 각 layer의 weight는 전부 평균이 0이고 표준편차가 0.1인 Gaussian Distribution의 값으로 초기화 하였다. bias의 경우 2,4,5번째 Conv layer와 FCN들은 1로, 나머지 layer들은 0으로 초기화 하였다. 이 Initialization은 early stage에서 학습이 잘 되도록 도와줬다.(ReLU를 쓰고 있기 때문에 positive input을 넣어 학습이 잘 되는 효과를 얻은 것이다.)
- 모든 layer에 대해 동일한 learning rate를 0.01로 줬다. 다른 연구에서도 쓰이던 방법인, validation error가 더 이상 하락하지 않을 때 learning rate를 10으로 나누는 방법을 사용했다. 지금까지의 방법을 통해, GTX 580 3GB짜리 GPU 2개로 1,200만 장의 train data를 학습하는 데 90 epoch이 소요되었고, 이는 5~6일 정도 소요되었다.

## 6. Results

- ILSVRC-2010에서 기존 다른 모델들을 제치고 Best score를 달성하였다.(Top-1: 37.5%, Top-5:17.0%)
![Untitled](https://user-images.githubusercontent.com/33994833/169384221-dbbf2d36-820f-4535-8391-b7eac76a5601.png)
- ILSVRC-2012에서도 기존 best 모델을 이기고 SOTA를 달성하였다. 1 CNN은 논문에서 소개한 모델을 의미하고, 5 CNN은 CNN 5개를 사용한 후 예측값을 평균 낸 경우이다.(Dropout의 존재로 인해 학습 결과가 매번 다를 것이기 때문에 모델 5개를 사용해서 괜찮은 성능이 나온 것 같다.) 그리고 추가로 1 CNN*은 ImageNet Fall 2011의 Dataset으로 학습된 모델에 6번째 Conv layer를 넣어 Fine-tuning한 모델로서 16.6%를 기록하였고, 7 CNNs*은 앞의 1 CNN* 구조의 모델 2개와 앞의 5 CNNs의 예측을 평균 낸 모델로서, Best score인 15.3%의 test Top-5 error rate를 기록하였다.

![Untitled](https://user-images.githubusercontent.com/33994833/169384382-11b68651-94b9-4abd-a947-42352c7c6482.png)
- Fall 2009 version of ImageNet(10,184 categories and 8.9 million imges)에도 모델을 실험해봤고, 위의 * 모델과 마찬가지로 마지막 pooling layer(5번째 Conv)이후에 6번째 Conv layer를 추가해 Fine-tuning 함으로서, top1-error 67.4%, top-5 error 40.9%를 달성하였다. 이는 기존의 best score인 78.1%와 60.9%를 뛰어넘는다. 참고로 전체 이미지를 반으로 나눠 반은 train, 반은 test data로 사용하였다.

### 6.1 Qualiative Evaluations

- 이 사진의 위쪽 48개 kernel은 1번째 GPU, 아래 48개 kernel은 2번째 GPU에서 Input data가 1번째 Conv layer를 통과하며 학습된 kernel들이다. 모델은 색상의 blob 뿐만 아니라 주파수(frequency), 방향(distance) 등의 특징을 학습하기도 했는데, GPU 간에 학습하는 특징에 차이가 있는 것을 확인할 수 있다.(GPU 1은 색상 관련 특징에 별로 구애받지 않으나, GPU 2는 구애받는 모습이다.) 이러한 결과는 매 run 마다 발생하고 GPU renumbering과는 무관하다고 했다.

![Untitled](https://user-images.githubusercontent.com/33994833/169384665-404ad709-abbb-4c5c-b851-f9430a205556.png)
- 아래 사진은 8장의 test image에 대해 real label과 Top-5 예측을 보여준다. 빨간색 막대가 real label을 의미한다. 이를 통해 mite 같이 object가 corner에 있는 경우도 예측을 잘 하는 것을 확인할 수 있다. 대부분의 Top-5 label들은 대체로 합리적인(그렇게 생각할 가능성이나 건덕지가 있는) 것으로 보이는데, 이는 leopard(표범)의 Top-5 label들이 재규어, 치타, 백표범, 이집트 고양이 인 것을 통해 표범과 비슷한 고양이과 생물들이 비교적 유사하다고 판단되고 있다. 예측이 잘 안된 case를 보면 사진이 애매모호하게(ambiguous) 된 경우가 있다.(grille, cherry) 

![Untitled](https://user-images.githubusercontent.com/33994833/169384792-39eef336-5db2-475c-9f9f-6356b36f71fb.png)
- 아래 사진의 왼쪽에서 첫 번째 Column은 Test data이고 오른쪽의 6장의 이미지들은 Test data와 비교했을 때, 각각 모델을 통과해 4096-dimension의 마지막 FCN을 통과한 직후의 vector가 가장 유사한(Euclidean distance가 가까운) 6장의 이미지를 나타낸 것이다. 이를 통해, pixel 레벨에선 테스트 이미지와 선택된 이미지 간의 L2 distance가 가깝지 않다는 것을 알 수 있는데, 개나 코끼리의 경우 다양한 포즈의 사진이 유사하다고 판단된 것을 통해서도 알 수 있다. 즉 단순 픽셀 유사도보다 좀 더 근본적인 유사도를 모델을 통과하며 얻은 feature activation과의 Euclidean distance로 알 수 있다.
- 두 4096-dimension의 vector간 Euclidean distance를 구할 때 실제 값 벡터(real-valued vector)를 통해 구하는 것은 비효울적이지만, Autoencoder를 통해 이 vector들을 짧은 binary code로 압축할 수 있게 학습시킬 수 있다.  Autoencoder를 raw-pixel에 적용하면, 이미지의 label을 활용할 수 없고 그렇기 때문에 의미적으로 비슷한 것 대신 이미지의 비슷한 edge pattern을 찾는 경향이 있는데, embedding 된 vector들을 적용하면 좀 더 의미적인 관점에서 이미지를 retrieve 할 수 있을 것이다.

![Untitled](https://user-images.githubusercontent.com/33994833/169384897-ff3f6fcf-fd87-4f0c-adc6-e38dc744c3d6.png)
## 7. Discussion

- 앞의 result를 통해, Deep convolutional neural network를 통해 고난이도 데이터셋에 대해 기록 경신을 할 정도로 좋은 성능을 낼 수 있다는 것을 증명했다. 다만 중간의 어느 Conv layer라도 없앨 경우 성능이 내려갔다는 점을 기억해야 한다. 실제로 아무 중간 Conv layer를 없애면 Top-1 error가 2% 올라갔다. 즉 Depth가 현재의 성능을 내는 데 중요하게 작용했다.
- 실험을 단순화하기 위해 이론상 가능함에도 불구하고 Unsupervised learning 기법을 사용하지 않았다. 그와 함께 앞으로 가야할 길이 많이 남았다며, 더 크고 깊은 네트워크를 통해 인간이 놓치기 쉬운, 도움 되는 정보를 추출하는 CNN을 만들고 싶다고 하며 논문을 마쳤다.
