---
title: 050 CNN모델의 발전과 전이 학습
tag:
  - 헬스케어 ai
  - deeplearing
  - tensorflow
description: 260706 수업 내용 정리
---

# CNN 모델의 발전과 Transfer Learning

이미지 인식 분야에서 CNN(Convolutional Neural Network)은 지난 십여 년간 놀라운 속도로 발전해왔다. 하지만 매번 새로운 문제를 만날 때마다 방대한 데이터로 모델을 처음부터 학습시키는 것은 현실적으로 비효율적이며, 이 문제를 해결하기 위해 등장한 것이 전이학습(Transfer Learning)이다. 여기서는 CNN의 성능을 끌어올려온 핵심 기법들을 먼저 살펴보고, 이어서 VGG-16부터 Inception, Xception, MobileNet, ResNet, EfficientNet에 이르는 대표적인 CNN 아키텍처들이 어떤 문제의식에서 출발해 어떻게 발전해왔는지를 순서대로 정리한다.

---

## 1. CNN 성능 개선의 두 가지 축

CNN의 성능 개선은 크게 두 가지 방향으로 이루어진다. 하나는 학습 과정 자체를 더 빠르고 정확하게 만드는 방향이고, 다른 하나는 네트워크의 구조 자체를 개선하는 방향이다.

학습을 고속화하고 정확하게 만드는 대표적인 요소로는 지식 재사용(전이학습), Batch Normalization, ReLU 활성화 함수, 그리고 Optimizer의 개선을 들 수 있다. 지식 재사용은 이미 학습된 모델의 가중치를 그대로 가져와 활용함으로써 학습 시간을 크게 단축하는 방법으로, 다음 절에서 자세히 다룬다.

Batch Normalization은 각 층의 입력값 분포를 미니배치 단위로 정규화하는 기법이다. 신경망이 깊어질수록 앞쪽 층의 파라미터가 조금만 바뀌어도 뒤쪽 층이 받는 입력의 분포가 크게 흔들리는 현상이 발생하는데, 이를 내부 공변량 변화(Internal Covariate Shift)라 부른다. Batch Normalization은 각 층의 출력을 평균 0, 분산 1로 정규화한 뒤 학습 가능한 스케일(`γ`)과 시프트(`β`) 파라미터를 다시 적용하여, 학습을 안정시키고 더 큰 학습률을 사용할 수 있게 해준다.

```
BN(x) = γ · (x - μ_batch) / sqrt(σ²_batch + ε) + β
```

여기서 `μ_batch`와 `σ²_batch`는 미니배치 내 평균과 분산이고, `ε`는 분모가 0이 되는 것을 막기 위한 아주 작은 상수다. `γ`와 `β`는 정규화된 값을 다시 원하는 스케일로 조정하기 위해 학습되는 파라미터로, 정규화 자체가 오히려 손해라면 네트워크가 스스로 원래 분포에 가깝게 복원할 수 있도록 여지를 남겨둔다.

```python
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU

x = Conv2D(64, (3, 3), padding='same')(x)
x = BatchNormalization()(x)   # 배치 단위로 정규화하여 학습 안정화
x = ReLU()(x)
```

ReLU(Rectified Linear Unit)는 `f(x) = max(0, x)`로 정의되는 활성화 함수다. 기존에 널리 쓰이던 Sigmoid나 Tanh는 입력값이 커지거나 작아질수록 기울기가 0에 가까워지는 기울기 소실(Vanishing Gradient) 문제가 심각했는데, ReLU는 양수 구간에서 기울기가 항상 1이기 때문에 이 문제를 크게 완화한다. 또한 연산 자체가 단순한 비교와 선택만으로 이루어져 계산 속도도 빠르다.

Optimizer 역시 SGD에서 시작해 Momentum, Adagrad, RMSprop, Adam 순으로 발전해왔다. SGD는 매 스텝마다 계산된 그래디언트 방향으로만 이동하기 때문에 진동이 심하고 수렴이 느린 반면, Adam은 그래디언트의 1차 모멘트(평균)와 2차 모멘트(분산)를 함께 추정하여 파라미터마다 학습률을 적응적으로 조절하기 때문에 대부분의 딥러닝 문제에서 안정적이고 빠른 수렴을 보인다.

네트워크 구조 개선 측면에서는 해상도(Resolution), 폭(Width), 깊이(Depth)라는 세 가지 방향이 있다. 해상도를 높이면 더 세밀한 정보를 담을 수 있고, 폭을 넓히면(채널 수를 늘리면) 한 층에서 더 다양한 특징을 동시에 추출할 수 있으며, 깊이를 늘리면 더 추상적이고 계층적인 표현을 학습할 수 있다. 이 세 가지를 어떻게 조합해야 가장 효율적인지에 대한 고민은 뒤에서 다룰 EfficientNet의 Compound Scaling 개념으로 이어진다. 이제 이러한 개선 방법들이 실제로 어떻게 적용되는지, 가장 기본이 되는 개념인 전이학습부터 살펴보자.

---

## 2. Transfer Learning의 개념과 접근법

전이학습은 ImageNet과 같은 대규모 데이터셋으로 이미 학습된 모델(pre-trained model)을 가져와, 풀고자 하는 새로운 문제에 맞게 재활용하는 기법이다. 사전학습된 모델의 앞쪽 층들은 이미 엣지, 색상, 질감 같은 저수준 특징과 형태 같은 중간 수준 특징을 잘 포착하고 있기 때문에, 이를 그대로 재사용하면 훨씬 적은 데이터와 훈련 시간으로도 좋은 성능을 얻을 수 있다. 전이학습을 적용하는 대표적인 방법은 다음 세 가지로 구분된다.

첫 번째는 Feature Extraction(특징 추출)이다. 사전학습된 모델의 합성곱 층은 그대로 동결(freeze)하여 가중치가 업데이트되지 않도록(`trainable = False`) 하고, 마지막 분류기(Dense/Output 층)만 새로운 문제에 맞게 교체해 학습하는 방식이다. 합성곱 층의 파라미터가 고정되어 있으므로 학습 속도가 매우 빠르고, 데이터가 적을 때 과적합 위험도 낮다.

```python
from tensorflow.keras.applications import VGG16
from tensorflow.keras import layers, models

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
base_model.trainable = False  # 합성곱 층 전체 동결 → feature extraction

model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(256, activation='relu'),
    layers.Dense(10, activation='softmax')  # 새로운 문제의 클래스 수
])
```

두 번째는 Fine-tuning(미세 조정)이다. Feature Extraction만으로 성능이 충분하지 않을 때, 동결했던 상위(출력에 가까운) 일부 층의 동결을 풀고 매우 작은 학습률로 함께 재학습시키는 방식이다. 입력에 가까운 하위 층은 범용적인 특징을 담고 있어 계속 동결하고, 출력에 가까운 상위 층만 새로운 데이터셋의 특성에 맞게 조정한다. 학습률을 작게 설정하는 이유는 이미 잘 학습된 가중치가 급격한 업데이트로 인해 손상되는 것을 막기 위해서다.

```python
base_model.trainable = True
for layer in base_model.layers[:-4]:
    layer.trainable = False  # 하위 층은 계속 동결

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),  # 아주 작은 학습률
              loss='categorical_crossentropy', metrics=['accuracy'])
```

세 번째는 LoRA(Low-Rank Adaptation)다. LoRA는 원본 가중치 행렬 `W`를 그대로 고정한 채, 훨씬 작은 크기의 저랭크(low-rank) 행렬 두 개(`A`, `B`)를 추가로 붙여 `W + BA` 형태로 극히 일부 파라미터만 학습시키는 기법이다. 전체 가중치를 다시 학습하는 것보다 메모리와 연산량을 크게 절약하면서도 파인튜닝에 준하는 성능을 낼 수 있어, 초대형 언어모델이나 대형 비전 모델을 가볍게 커스터마이징할 때 널리 쓰인다.

세 가지 방식은 다음과 같이 비교할 수 있다.

|방식|학습 대상|필요 데이터량|학습 속도|적합한 상황|
|---|---|---|---|---|
|Feature Extraction|새로 추가한 분류기만|적음|매우 빠름|새 데이터가 사전학습 데이터와 유사할 때|
|Fine-tuning|상위 일부 층 + 분류기|중간|보통|새 데이터가 어느 정도 다르고 데이터가 충분할 때|
|LoRA|저랭크 보조 행렬만|적음~중간|빠름(파라미터 효율적)|초대형 모델을 가볍게 커스터마이징할 때|

전이학습이라는 큰 틀을 이해했다면, 이제 실제로 전이학습의 기반이 되는 대표적인 사전학습 모델들이 각각 어떤 구조적 아이디어로 만들어졌는지 시대 순으로 살펴보자.

---

## 3. VGG-16

VGG-16은 옥스포드 대학 연구팀이 발표한 모델로, 작은 필터를 여러 번 쌓으면 큰 필터 하나보다 효율적이라는 원칙을 증명한 대표적인 구조다. 이름의 16은 학습 가능한 가중치를 가진 층, 즉 합성곱 13개와 Dense 3개를 합한 개수를 의미한다.

VGG-16의 핵심 설계 원칙은 모든 합성곱 필터 크기를 3x3으로 통일하고, 패딩은 SAME을 사용해 합성곱을 거쳐도 특징 맵의 가로·세로 크기가 줄어들지 않게 하며, 크기 축소는 오직 Pooling 층에서만 담당하도록 한 것이다.

```
Input → [Conv1-1, Conv1-2] → Pooling
      → [Conv2-1, Conv2-2] → Pooling
      → [Conv3-1, Conv3-2, Conv3-3] → Pooling
      → [Conv4-1, Conv4-2, Conv4-3] → Pooling
      → [Conv5-1, Conv5-2, Conv5-3] → Pooling
      → Dense → Dense → Output
```

3x3 필터를 두 번 연속으로 쌓으면 5x5 필터 한 개와 동일한 수용 영역(receptive field)을 확보하면서도, 파라미터 수는 훨씬 적고(채널 수를 고려하면 차이가 더 커짐) 그 사이에 비선형 활성화 함수를 한 번 더 적용할 수 있어 표현력도 함께 좋아진다. 이것이 3x3 필터와 SAME 패딩으로 빠르고 정보 손실 없이 학습한다는 설계 의도의 핵심이다.

```python
from tensorflow.keras.applications import VGG16

model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
model.summary()
```

VGG-16은 구조가 단순해 이해와 구현이 쉽지만, 파라미터 수가 약 1억 3천만 개에 달해 메모리 사용량이 크고 층이 깊어질수록 기울기 소실 문제에 취약하다는 한계가 있다. 이 한계를 해결하기 위해 이후 Inception과 같은 새로운 구조가 등장하게 된다.

---

## 4. Inception v4

Inception 계열 모델의 핵심 아이디어는 합성곱 필터의 크기를 하나로 고정하지 않고, 여러 크기를 병렬로 적용한 뒤 그 결과를 하나로 합치는 것이다. Inception v4는 이 아이디어를 정교하게 다듬은 버전으로, 하나의 Inception 모듈은 이전 층의 출력을 입력받아 1x1, 3x3, 5x5 합성곱과 3x3 max pooling을 병렬로 적용한 뒤 그 결과들을 채널 방향으로 이어붙이는 구조를 갖는다.

```
                Previous layer
        ┌───────────┼───────────┬───────────┐
     1x1 conv     1x1 conv    3x3 max pool  1x1 conv
        │            │            │
     3x3 conv     5x5 conv     1x1 conv
        └───────────┴───────────┴───────────┘
                Filter concatenation
```

여기서 3x3, 5x5 합성곱 앞에 놓인 1x1 합성곱은 차원 축소(dimension reduction) 역할을 한다. 예를 들어 채널 수가 256인 입력에 곧바로 5x5 합성곱을 적용하면 연산량이 매우 크지만, 먼저 1x1 합성곱으로 채널 수를 64 정도로 줄인 뒤 5x5 합성곱을 적용하면 연산량을 크게 절감하면서도 정보 손실은 최소화할 수 있다. 이렇게 여러 크기의 필터로 얻은 결과를 채널 방향으로 이어붙이는 과정이 Filter Concatenation이며, 이 구조 덕분에 이미지 안의 객체 크기가 제각각이어도 작은 필터는 작은 객체를, 큰 필터는 큰 객체를 동시에 잘 포착할 수 있게 된다.

Inception이 여러 크기의 필터를 병렬로 두어 문제를 해결했다면, 다음으로 살펴볼 Xception은 이 병렬 구조를 오히려 더 단순한 연산으로 대체하는 방향으로 나아간다.

---

## 5. Xception

Xception은 Extreme Inception의 줄임말로, Inception 모듈의 아이디어를 극단까지 밀어붙여 Depthwise Separable Convolution이라는 새로운 연산으로 대체한 모델이다.

일반적인 합성곱은 하나의 필터가 모든 입력 채널을 한 번에 보고 하나의 출력 값을 계산한다. 반면 Depthwise Separable Convolution은 이 과정을 두 단계로 분리한다. 먼저 Depthwise Convolution 단계에서는 각 입력 채널에 대해 독립적으로 하나씩 필터를 적용하여, 채널 간 정보를 섞지 않고 채널 각각의 공간적(spatial) 특징만 추출한다. 이어서 Pointwise Convolution 단계에서는 1x1 합성곱을 사용해 앞 단계 결과를 채널 방향으로 결합(mixing)한다.

```
입력 (채널 수 C)
   ↓
[Depthwise Convolution]  ← 채널별로 독립적인 공간 필터링
   ↓
[Pointwise Convolution]  ← 1x1 conv로 채널 간 정보 결합
   ↓
출력
```

이렇게 공간 방향 필터링과 채널 방향 결합을 분리하면 연산량과 파라미터 수를 크게 줄일 수 있다. 입력 채널 수를 `Cin`, 출력 채널 수를 `Cout`, 필터 크기를 `k×k`라 할 때, 일반 합성곱의 연산량은 대략 `k² × Cin × Cout`에 비례하지만, Depthwise Separable Convolution은 `k² × Cin + Cin × Cout`에 비례하여 `Cout`이 클수록 절감 효과가 더욱 커진다.

```python
from tensorflow.keras.layers import SeparableConv2D

x = SeparableConv2D(filters=64, kernel_size=(3, 3), padding='same', activation='relu')(x)
```

Xception이 제안한 이 연산은 곧 모바일 환경을 겨냥한 경량 모델의 표준 구성 요소로 자리 잡는데, 그 대표 사례가 MobileNet이다.

---

## 6. MobileNet

MobileNet은 Xception에서 제안된 Depthwise Separable Convolution을 모바일 기기처럼 연산 자원이 제한된 환경에서도 실시간으로 동작할 수 있도록 경량화한 모델이다. 기본 블록은 3x3 Depthwise Convolution 뒤에 Batch Normalization과 ReLU를 적용하고, 이어서 1x1 Pointwise Convolution 뒤에 다시 Batch Normalization과 ReLU를 적용하는 구조로 이루어진다.

```
입력
 ↓
[3x3 Depthwise Convolution] → [Batch Normalization] → [ReLU]
 ↓
[1x1 Pointwise Convolution] → [Batch Normalization] → [ReLU]
 ↓
출력
```

Xception과 원리는 동일하지만, MobileNet은 각 합성곱 뒤에 Batch Normalization과 ReLU를 명시적으로 붙여 학습 안정성과 비선형성을 강화한 점이 특징이다. 또한 `width multiplier`와 `resolution multiplier`라는 하이퍼파라미터를 통해 모델의 크기와 연산량을 필요에 따라 조절할 수 있어, 서버용 고정확도 모델부터 초경량 모바일 모델까지 유연하게 대응할 수 있다.

```python
from tensorflow.keras.applications import MobileNetV2

model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), alpha=0.75)
# alpha(width multiplier)를 1보다 작게 설정하면 채널 수가 줄어들어 더 가벼운 모델이 됨
```

지금까지 살펴본 Inception, Xception, MobileNet은 모두 하나의 층 내부에서 연산을 어떻게 효율화할지에 초점을 맞춘 구조였다. 이제 관점을 바꿔, 네트워크를 아주 깊게 쌓았을 때 발생하는 문제와 그 해결책을 살펴보자.

---

## 7. ResNet

네트워크를 깊게 쌓을수록 이론적으로는 더 복잡한 함수를 학습할 수 있어야 하지만, 실제로는 층이 지나치게 깊어지면 기울기 소실 문제로 인해 학습이 오히려 잘 되지 않고 성능이 떨어지는 성능 저하(degradation) 문제가 발생한다. ResNet(Residual Neural Network)은 이 문제를 Skip Connection(잔차 연결)이라는 아이디어로 해결했다.

일반적인 층은 입력 `x`로부터 목표 함수 `H(x)`를 직접 학습하려 하지만, ResNet은 다음과 같이 목표를 재정의한다.

```
출력 = F(x) + x
```

여기서 `F(x)`는 두 개의 가중치 층과 ReLU로 구성된 잔차 함수(residual function)이고, `x`는 입력을 그대로 더해주는 identity mapping(항등 매핑)이다.

```
     x
     │
[weight layer]
     │
   [ReLU]
     │
[weight layer]
     │
     ├───────── x (identity)
     ↓          │
   F(x) + x ←───┘
     │
   [ReLU]
```

이 구조가 학습을 쉽게 만드는 이유는, 만약 항등 함수(`H(x) = x`)가 최적이라면 `F(x)`가 0에 가까워지도록만 학습하면 되기 때문이다. 0을 학습하는 것은 임의의 복잡한 항등 함수를 직접 학습하는 것보다 훨씬 쉽다. 또한 역전파 시 그래디언트가 identity 경로를 통해 그대로 하위 층까지 전달될 수 있어, 기울기 소실 문제가 크게 완화된다. 이 원리 덕분에 ResNet은 152개 층(ResNet-152)까지 매우 깊게 쌓으면서도 안정적으로 학습이 가능함을 보여주었고, 이후 등장한 EfficientNet 등 다른 아키텍처들도 Skip Connection을 기본 구성 요소로 채택하게 된다.

```python
from tensorflow.keras import layers

def residual_block(x, filters):
    shortcut = x  # identity 경로
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Add()([x, shortcut])  # F(x) + x
    x = layers.ReLU()(x)
    return x
```

ResNet이 깊이(Depth)의 한계를 뚫는 방법을 제시했다면, 이 Skip Connection의 아이디어를 조금 다른 방식으로 극단까지 밀어붙인 구조가 DenseNet이다.

---

## 8. DenseNet

ResNet의 Skip Connection은 이전 블록의 출력을 뒤쪽 층에 그대로 더해주는 방식(`F(x) + x`)이었다. DenseNet(Densely Connected Convolutional Network)은 여기서 한 걸음 더 나아가, "각 층이 자신보다 앞에 있는 모든 층의 출력을 입력으로 받는다"는 아이디어를 제시한다. 덧셈이 아니라 채널 방향으로 이어붙이는 연결(concatenation) 방식을 사용한다는 점이 ResNet과의 가장 큰 차이다.

`l`번째 층의 입력을 수식으로 표현하면 다음과 같다.

```
x_l = H_l([x_0, x_1, x_2, ..., x_{l-1}])
```

여기서 `[x_0, x_1, ..., x_{l-1}]`은 0번째 층부터 `l-1`번째 층까지의 출력을 채널 방향으로 이어붙인 것을 의미하고, `H_l`은 Batch Normalization, ReLU, 3x3 Convolution으로 구성된 합성 함수(composite function)다. 즉 5번째 층은 0, 1, 2, 3, 4번째 층의 출력을 모두 입력으로 받아 사용하며, ResNet이 바로 이전 블록만 참조하는 것과 달리 훨씬 더 촘촘한(densely connected) 연결 구조를 이룬다.

이 구조를 실제로 구현하려면 채널 수가 계속 누적되어 폭발적으로 증가하는 문제를 관리해야 한다. DenseNet은 이를 위해 두 가지 장치를 도입한다. 첫 번째는 Growth Rate(`k`)로, 각 층이 생성하는 새로운 특징 맵의 수를 `k`개로 제한한다. 예를 들어 `k=32`이고 입력 채널이 64였다면, 첫 번째 층을 통과한 뒤 채널 수는 `64 + 32 = 96`이 되고, 두 번째 층을 통과하면 `96 + 32 = 128`이 되는 식으로 선형적으로 증가한다. 두 번째는 Transition Layer로, 하나의 Dense Block이 끝날 때마다 Batch Normalization, 1x1 Convolution, 2x2 Average Pooling을 두어 채널 수를 압축하고 특징 맵의 가로·세로 크기를 절반으로 줄인다.

```python
from tensorflow.keras import layers

def dense_block(x, num_layers, growth_rate):
    for _ in range(num_layers):
        y = layers.BatchNormalization()(x)
        y = layers.ReLU()(y)
        y = layers.Conv2D(growth_rate, 3, padding='same')(y)
        x = layers.Concatenate()([x, y])  # 이전 모든 층의 출력을 계속 이어붙임
    return x

def transition_layer(x, compression=0.5):
    filters = int(x.shape[-1] * compression)
    x = layers.BatchNormalization()(x)
    x = layers.Conv2D(filters, 1, padding='same')(x)  # 채널 압축
    x = layers.AveragePooling2D(pool_size=2, strides=2)(x)  # 크기 절반으로 축소
    return x
```

DenseNet이 갖는 장점은 크게 세 가지다. 첫째, 모든 층이 앞선 층들의 특징을 직접 참조할 수 있어 특징 재사용(feature reuse)이 극대화되고 같은 정보를 중복으로 다시 학습할 필요가 줄어든다. 둘째, 각 층이 입력에 가까운 층까지 직접 연결되어 있어 그래디언트가 더 짧은 경로로 전달되므로, 기울기 소실 문제가 ResNet보다도 더 효과적으로 완화된다. 셋째, 역설적으로 각 층이 적은 수의 새로운 특징만 생성하면 되기 때문에, 비슷한 성능을 내는 ResNet보다 오히려 전체 파라미터 수가 적어지는 경우가 많다. 다만 중간 특징 맵을 계속 보관해야 하므로 학습 시 메모리 사용량은 ResNet보다 커질 수 있다는 점은 함께 고려해야 한다.

ResNet과 DenseNet은 모두 Skip Connection이라는 같은 뿌리에서 출발했지만, 정보를 결합하는 방식에 따라 특성이 뚜렷하게 갈린다.

|구분|ResNet|DenseNet|
|---|---|---|
|결합 방식|덧셈 (`F(x) + x`)|연결 (`concat([x_0, ..., x_{l-1}])`)|
|연결 범위|바로 이전 블록의 출력만|같은 Dense Block 내 모든 이전 층의 출력|
|채널 수 변화|블록을 지나도 유지|층을 지날 때마다 growth rate만큼 증가|
|파라미터 효율성|상대적으로 많음|특징 재사용으로 상대적으로 적음|
|메모리 사용량|상대적으로 적음|중간 특징 맵을 계속 보관해 상대적으로 큼|

ResNet과 DenseNet이 깊이 방향의 문제를 해결하는 두 가지 답을 보여주었다면, 마지막으로 살펴볼 EfficientNet은 깊이뿐 아니라 폭과 해상도까지 함께 고려하는 통합적인 관점을 제시한다.

---

## 9. EfficientNet

앞서 언급한 세 가지 확장 방향인 깊이(Depth), 폭(Width), 해상도(Resolution) 중 기존 모델들은 보통 하나만 확대해서 성능을 높여왔다. 폭만 넓히거나, 깊이만 늘리거나, 해상도만 높이는 방식이다. 그러나 EfficientNet 논문은 이 셋 중 하나만 키우면 일정 수준 이상에서 성능 향상이 빠르게 정체된다는 것을 실험적으로 보였다. 예를 들어 해상도만 높이면 그에 맞는 더 넓고 깊은 네트워크가 뒷받침되지 않아 세밀해진 정보를 충분히 활용하지 못한다.

그래서 제안된 것이 Compound Scaling으로, 깊이·폭·해상도를 하나의 계수 `φ(phi)`로 동시에, 정해진 비율에 따라 함께 확장한다.

```
depth:      d = α^φ
width:      w = β^φ
resolution: r = γ^φ

제약조건: α · β² · γ² ≈ 2,   α ≥ 1, β ≥ 1, γ ≥ 1
```

여기서 `α, β, γ`는 소규모 그리드 서치로 찾은 고정 상수이고, `φ`는 사용 가능한 연산 자원에 따라 조절하는 값이다. `φ`를 0부터 순차적으로 늘려가며 만든 모델들이 EfficientNet-B0부터 B7까지이며, B0가 가장 작고 B7로 갈수록 깊이·폭·해상도가 모두 커지면서 정확도와 연산량이 함께 증가한다.

```python
from tensorflow.keras.applications import EfficientNetB0

model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
```

지금까지 살펴본 아키텍처들을 정리하면 다음과 같다.

|모델|핵심 아이디어|주요 장점|주요 단점|
|---|---|---|---|
|VGG-16|3x3 필터를 균일하게 깊게 쌓음|구조가 단순하고 이해하기 쉬움|파라미터 수가 매우 많음|
|Inception v4|여러 크기 필터를 병렬 적용 후 결합|다양한 스케일의 특징을 동시에 포착|구조가 복잡하고 설계가 어려움|
|Xception|Depthwise Separable Convolution|연산량·파라미터 수 대폭 절감|채널 간 상호작용이 다소 제한적|
|MobileNet|경량화된 Depthwise Separable Conv|모바일·임베디드에 적합|대규모 서버 환경에서는 정확도 한계|
|ResNet|Skip Connection(잔차 연결, 덧셈 방식)|매우 깊은 네트워크 학습 가능|깊이에 비례해 메모리 사용량 증가|
|DenseNet|Dense Connectivity(연결 방식 Skip Connection)|특징 재사용으로 파라미터 효율 우수|중간 특징 맵 저장으로 메모리 사용량 큼|
|EfficientNet|Compound Scaling(깊이·폭·해상도 동시 조절)|적은 연산량으로 높은 정확도 달성|스케일링 계수 탐색에 별도 비용 필요|

---

## 10. 핵심 요약 (Key Summary)

- CNN 성능 개선은 학습 방식 개선(전이학습, Batch Normalization, ReLU, Optimizer)과 네트워크 구조 개선(Resolution, Width, Depth)이라는 두 축으로 이루어진다.
- 전이학습은 Feature Extraction(합성곱 층 동결, 분류기만 학습), Fine-tuning(상위 일부 층까지 재학습), LoRA(저랭크 보조 행렬만 학습)로 구분되며 데이터량과 목적에 따라 선택한다.
- VGG-16은 3x3 필터와 SAME 패딩을 규칙적으로 쌓아 깊은 네트워크를 단순한 구조로 구현했다.
- Inception v4는 1x1/3x3/5x5 등 여러 크기의 필터를 병렬로 적용한 뒤 Filter Concatenation으로 결합하며, 1x1 합성곱으로 연산량을 줄인다.
- Xception과 MobileNet은 Depthwise Convolution(채널별 공간 필터링)과 Pointwise Convolution(1x1로 채널 결합)을 사용해 연산량을 크게 줄인다.
- ResNet은 `출력 = F(x) + x` 형태의 Skip Connection(Residual)을 도입해 매우 깊은 네트워크에서도 기울기 소실 없이 안정적으로 학습할 수 있게 했다.
- DenseNet은 `x_l = H_l([x_0, ..., x_{l-1}])` 형태로 같은 Dense Block 내 모든 이전 층의 출력을 채널 방향으로 이어붙이며, Growth Rate와 Transition Layer로 채널 수 증가를 관리해 특징 재사용과 파라미터 효율을 동시에 확보했다.
- EfficientNet은 깊이·폭·해상도를 각각 따로 키우지 않고 하나의 계수 `φ`로 동시에 균형 있게 확장하는 Compound Scaling을 통해 B0~B7까지 효율적인 모델 라인업을 완성했다.