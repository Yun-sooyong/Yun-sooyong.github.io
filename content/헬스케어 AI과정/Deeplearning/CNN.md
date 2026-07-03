---
title: 049 CNN
tag:
  - 헬스케어 ai
  - deeplearing
  - tensorflow
description: 260703 수업 내용 정리
---

# CNN (Convolutional Neural Network)

CNN은 이미지, 영상, 음성처럼 **공간적 구조를 가진 데이터**를 처리하는 데 특화된 신경망이다. MLP(완전 연결망)가 모든 픽셀을 같은 선상의 숫자로 펼쳐서 처리하는 반면, CNN은 픽셀들의 **공간적 위치 관계**를 유지하면서 처리한다.

```
MLP 방식 (Flatten):
  6×6 이미지 → 36개 숫자로 펼침 → Dense(128) → Dense(64) → 예측
  → 위아래 픽셀의 관계, 좌우 픽셀의 관계를 모두 잃어버림

CNN 방식:
  6×6 이미지 → 그대로 유지 → 필터로 지역 패턴 감지 → 예측
  → "오른쪽 위 부분에 귀처럼 생긴 패턴이 있다"는 정보 유지
```

이 글에서는 CNN을 이해하는 데 필요한 핵심 개념들(채널, 합성곱, 풀링, 패딩, 스트라이드)을 먼저 살펴보고, 실제 모델 구조와 학습 과정, 그리고 모델 해석 방법까지 순서대로 다룬다.

---

## 1. 이미지 데이터의 구조 — Channel

디지털 이미지는 3차원 배열이다. **가로 × 세로 × 채널(Channel)** 로 표현한다.

```
RGB 이미지 예시 (6×6 픽셀):
  Shape = (6, 6, 3)
           ↑   ↑  ↑
          세로 가로 채널(R, G, B)

  Red   채널: 각 픽셀의 빨간색 강도 (0~255)
  Green 채널: 각 픽셀의 초록색 강도
  Blue  채널: 각 픽셀의 파란색 강도

흑백(Grayscale) 이미지: Shape = (H, W, 1)   → 채널 1개
컬러(RGB) 이미지:        Shape = (H, W, 3)   → 채널 3개
RGBA 이미지 (투명도):    Shape = (H, W, 4)   → 채널 4개
위성 이미지:              Shape = (H, W, 8+)  → 다중 스펙트럼
```

### CNN 입력 형태

CNN 모델에 데이터를 넣을 때의 형태:

```
Input Shape = (배치 크기, 가로, 세로, 채널)
Input Shape = (N, W, H, C)

예: 32개의 64×64 RGB 이미지 배치
  → Shape = (32, 64, 64, 3)
       ↑     ↑   ↑   ↑
     배치  가로 세로 RGB

* TensorFlow/Keras는 채널이 마지막 (channels_last, 기본값)
* PyTorch는 채널이 세 번째 (N, C, H, W)
```

---

## 2. Convolution 연산

이미지의 구조를 이해했다면, 이제 CNN이 그 구조에서 패턴을 감지하는 핵심 연산인 합성곱을 살펴보자.

CNN의 핵심이다. 작은 필터(커널)가 이미지를 훑으면서 지역적 패턴을 감지한다.

### 원리

```
이미지 위에 작은 필터를 올려놓고:
  1. 필터와 겹치는 이미지 영역을 원소별로 곱함
  2. 모두 더함 (내적)
  3. 결과 하나의 숫자 생성
  4. 필터를 옆으로 한 칸 이동 (Stride만큼)
  5. 전체 이미지를 다 훑을 때까지 반복

→ 필터 하나 = 특정 패턴을 감지하는 탐지기
  수평선 감지 필터, 수직선 감지 필터, 대각선 감지 필터 ...
```

### 수치 예시

구체적인 숫자로 합성곱 연산을 따라가 보자.

```
입력 이미지 (5×5):         필터 (2×2):
2  1  2  0  1               2  3
3  2  2  3  2               0  1
1  1  3  3  0
1  1  1  3  0
0  0  3  1  2

필터를 왼쪽 위 (0,0) 위치에 놓으면:
  2×2 + 1×3 + 3×0 + 2×1 = 4 + 3 + 0 + 2 = 9 (아니라 10)

실제 계산:
  겹치는 영역: [[2,1],[3,2]]
  필터:        [[2,3],[0,1]]
  원소곱의 합: 2×2 + 1×3 + 3×0 + 2×1 = 4+3+0+2 = 9

  실제 계산: 2·2+3·(-1)+0·1+1·(-3) = 10 
  → 필터 값에 따라 결과가 달라짐

출력 (4×4):
10   6   6
12  15  13  13
11   7  11   6   7
10   7   4   7
```

출력 크기 계산 공식:

$$\text{출력 크기} = \frac{입력 크기 - 필터 크기}{Stride} + 1$$

```
입력 5×5, 필터 2×2, Stride 1이면:
  출력 크기 = (5 - 2) / 1 + 1 = 4   → 4×4 출력

입력 5×5, 필터 3×3, Stride 1이면:
  출력 크기 = (5 - 3) / 1 + 1 = 3   → 3×3 출력
```

### Filter (Multi Filter)

하나의 필터는 하나의 특징만 감지한다. 여러 필터를 동시에 사용하면 다양한 특징을 한 번에 감지할 수 있다.

```
필터 1: 수평선 감지
필터 2: 수직선 감지
필터 3: 대각선 감지
필터 4: 색상 경계 감지
...

필터 32개 → 출력 채널 32개 (각 필터가 만든 특징 맵 32장)
필터 64개 → 출력 채널 64개

Conv2D(32, kernel_size=3) 의미:
  3×3 필터 32개를 동시에 학습
  → 32가지 서로 다른 패턴을 감지하는 탐지기 32개
```

```python
import tensorflow as tf
from tensorflow.keras import layers

# Conv2D 파라미터 설명
conv_layer = layers.Conv2D(
    filters=32,            # 필터(커널) 수 = 출력 채널 수
    kernel_size=(3, 3),    # 필터 크기 (3×3)
    strides=(1, 1),        # 이동 보폭
    padding='same',        # 'same': 출력 크기 = 입력 크기 / 'valid': 크기 줄어듦
    activation='relu',     # 합성곱 후 활성화 함수
    kernel_initializer='he_normal',
    input_shape=(64, 64, 3)
)

# 입력/출력 shape 확인
x = tf.random.normal([1, 64, 64, 3])   # 배치 1, 64×64, RGB
out = conv_layer(x)
print(f'입력 shape: {x.shape}')        # (1, 64, 64, 3)
print(f'출력 shape: {out.shape}')      # (1, 64, 64, 32) — same padding

# 파라미터 수 계산:
# 필터 크기 × 입력 채널 × 출력 채널 + 편향
# 3 × 3 × 3 × 32 + 32 = 896
print(f'파라미터 수: {conv_layer.count_params()}')  # 896
```

---

## 3. Stride

합성곱 연산에서 필터가 한 번에 얼마나 이동하는지를 결정하는 것이 스트라이드다.

필터가 한 번에 이동하는 **보폭(칸 수)** 이다.

```
Stride = 1 (기본값):
  ┌─────────────────┐
  │ □ □ □ □ □      │
  │ □ □ □ □ □      │  필터가 한 칸씩 이동
  │ □ □ □ □ □      │
  └─────────────────┘
  → 출력 크기가 크게 유지됨

Stride = 2:
  ┌─────────────────┐
  │ □   □   □      │
  │                 │  필터가 두 칸씩 이동 (듬성듬성)
  │ □   □   □      │
  └─────────────────┘
  → 출력 크기가 절반으로 줄어듦

Stride가 크면:
  - 출력 크기 ↓ (다운샘플링 효과)
  - 연산량 ↓
  - 정보 손실 ↑
```

---

## 4. Padding

스트라이드와 합성곱을 적용하면 출력 크기가 줄어든다. 이것을 제어하고 이미지 가장자리 정보를 보존하기 위한 장치가 패딩이다.

합성곱 후 출력 크기가 줄어드는 것을 방지하거나, 이미지 가장자리 정보를 보존하기 위해 **이미지 주변에 값(보통 0)을 추가**하는 것이다.

```
패딩 없음 (valid):
  5×5 입력 + 3×3 필터 → 3×3 출력 (크기 줄어듦)

패딩 추가 (same):
  이미지 주변에 0을 추가해서 출력 크기 = 입력 크기 유지
  
  0 0 0 0 0 0 0
  0 2 1 2 0 1 0
  0 3 2 2 3 2 0    ← 원본 5×5 이미지 주변에 0 추가
  0 1 1 3 3 0 0
  0 1 1 1 3 0 0
  0 0 0 3 1 2 0
  0 0 0 0 0 0 0

  5×5 입력 + 패딩(1) → 7×7 → 3×3 필터 → 5×5 출력 (크기 유지!)
```

```python
# padding='valid': 출력 크기 줄어듦 (기본)
# padding='same':  출력 크기 = 입력 크기 (제로 패딩 자동 추가)

x = tf.random.normal([1, 5, 5, 1])

conv_valid = layers.Conv2D(1, 3, padding='valid')
conv_same  = layers.Conv2D(1, 3, padding='same')

print(conv_valid(x).shape)  # (1, 3, 3, 1) — 크기 줄어듦
print(conv_same(x).shape)   # (1, 5, 5, 1) — 크기 유지
```

---

## 5. Pooling

합성곱으로 특징을 추출했다면, 이제 불필요한 공간 정보를 줄이고 핵심만 남기는 과정이 필요하다. 이것이 풀링이다.

합성곱 후 특징 맵의 크기를 줄여서 **계산량을 낮추고 위치 변화에 강인하게** 만드는 연산이다. 가장 중요한 정보만 남기는 다운샘플링 역할을 한다.

### Max Pooling

지정한 영역에서 **가장 큰 값**만 선택한다. 특정 특징이 어디에 있는지 정확한 위치보다 "있냐 없냐"를 중시하는 효과가 있다.

```
Max Pooling 계산 예시:

합성곱 출력 (4×4):       Max Pool 2×2 (Stride=2):
6   6                     15  13
12  15  13  13    →   
11   7  11   6             □   □
10   7   4   7

2×2 영역에서 최댓값만 선택:
  왼쪽 위 2×2: max(6,6,12,15)   = 15
  오른쪽 위 2×2: max(6,13,13,?)  = ?
  
  → 출력: 2×2 (4×4에서 절반으로 축소)

```

### Average Pooling

지정한 영역의 **평균값**을 선택한다. 특징의 전반적인 강도를 보존한다. Max Pooling보다 부드러운 다운샘플링이다.

```python
# Max Pooling
max_pool = layers.MaxPooling2D(
    pool_size=(2, 2),  # 2×2 영역에서 최댓값
    strides=(2, 2),    # 2칸씩 이동 (일반적으로 pool_size와 같게 설정)
    padding='valid'
)

# Average Pooling
avg_pool = layers.AveragePooling2D(
    pool_size=(2, 2),
    strides=(2, 2)
)

# Global Average Pooling (GAP)
# 전체 특징 맵의 채널별 평균 → 벡터 하나로 압축
# Flatten 대신 사용하면 파라미터 수 대폭 감소
gap = layers.GlobalAveragePooling2D()

# 예시
x = tf.random.normal([1, 32, 32, 64])
print(max_pool(x).shape)  # (1, 16, 16, 64)
print(gap(x).shape)       # (1, 64)  ← 각 채널의 평균값 하나씩
```

### Max Pooling vs Average Pooling

|구분|Max Pooling|Average Pooling|
|---|---|---|
|선택 방식|영역 내 최댓값|영역 내 평균값|
|특징 보존|가장 강한 특징|전반적인 특징|
|잡음 제거|강함|약함|
|사용 위치|일반 은닉층|마지막 특징 추출 (GAP)|

---

## 6. CNN 전체 구조

지금까지 살펴본 채널, 합성곱, 스트라이드, 패딩, 풀링이 어떻게 하나의 모델로 조합되는지 전체 구조를 통해 확인해보자.

```
입력 이미지 (고양이 사진)
        ↓
  Convolution 1    ← 저수준 특징 (엣지, 색상 경계)
        ↓
  Max Pooling 1    ← 공간 크기 축소 (예: 64×64 → 32×32)
        ↓
  Convolution 2    ← 중간 특징 (눈, 코, 귀 모양)
        ↓
  Max Pooling 2    ← 다시 공간 크기 축소 (32×32 → 16×16)
        ↓
  마지막 특징맵
        ↓
  GAP or Flatten   ← 2D 특징맵을 1D 벡터로 변환
        ↓
  Fully-Connected (Dense)  ← 고수준 추상 특징 조합
        ↓
  Logit (Softmax 전)       ← 클래스별 점수
        ↓
  Softmax                  ← 확률 분포 출력
        ↓
  "cat: 0.92, dog: 0.05, ..."
```

층이 깊어질수록 학습하는 특징이 점점 추상화된다.

```
얕은 층: 엣지, 선, 색상 경계 (단순 패턴)
중간 층: 눈, 코, 귀, 발톱 (부분 객체)
깊은 층: 얼굴, 몸통 전체 (고수준 개념)
```

### TensorFlow/Keras로 구현

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# CNN 모델 구성
def build_cnn(input_shape=(64, 64, 3), num_classes=10):
    model = models.Sequential([
        # 1번째 Convolution 블록
        layers.Conv2D(32, (3,3), padding='same', activation='relu',
                      kernel_initializer='he_normal',
                      input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),      # 64×64 → 32×32
        layers.Dropout(0.25),

        # 2번째 Convolution 블록
        layers.Conv2D(64, (3,3), padding='same', activation='relu',
                      kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), padding='same', activation='relu'),
        layers.MaxPooling2D((2,2)),      # 32×32 → 16×16
        layers.Dropout(0.25),

        # 3번째 Convolution 블록
        layers.Conv2D(128, (3,3), padding='same', activation='relu',
                      kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),      # 16×16 → 8×8
        layers.Dropout(0.25),

        # 분류 헤드
        layers.GlobalAveragePooling2D(),  # (8,8,128) → (128,) — GAP
        layers.Dense(256, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')  # 다중 분류
    ], name='CNN')

    return model

model = build_cnn(input_shape=(64, 64, 3), num_classes=10)
model.summary()

# 파라미터 수 확인
total_params = model.count_params()
print(f'전체 파라미터: {total_params:,}개')
```

---

## 7. Flatten vs Global Average Pooling (GAP)

합성곱 층을 통과하면 3D 형태의 특징맵이 만들어진다. 이것을 최종 분류를 담당하는 Dense 층에 연결하려면 1D 벡터로 변환해야 한다. 두 가지 방법이 있으며, 선택에 따라 파라미터 수와 모델 성능이 크게 달라진다.

### Flatten

모든 값을 그냥 일렬로 펼친다.

```
(8, 8, 128) 특징맵 → Flatten → (8×8×128,) = (8192,) 벡터

다음 Dense(256) 연결 시 파라미터:
  8192 × 256 + 256 = 2,097,408개 (200만 개!)
  → 과적합 위험, 메모리 많이 필요
```

### Global Average Pooling (GAP)

각 채널(필터)마다 전체 특징맵의 평균을 내서 채널 수만큼의 벡터를 만든다. 쉽게 말해 "이 필터가 이미지 전체에서 평균적으로 얼마나 활성화됐는가"를 하나의 숫자로 요약하는 것이다.

```
(8, 8, 128) 특징맵:
  채널 0: 8×8 = 64개 값 → 평균 → 숫자 1개
  채널 1: 8×8 = 64개 값 → 평균 → 숫자 1개
  ...
  채널 127: 8×8 = 64개 값 → 평균 → 숫자 1개

  → (128,) 벡터 완성

다음 Dense(256) 연결 시 파라미터:
  128 × 256 + 256 = 32,896개 (약 3만 개)
  → Flatten 대비 60배 이상 파라미터 감소 → 과적합 방지에 유리
```

**GAP를 쓰는 이유**를 Flatten과 비교하면 더 명확해진다.

Flatten은 특징맵의 모든 값을 일렬로 펼치기 때문에 이미지 크기에 따라 벡터 길이가 크게 달라진다. 8×8 특징맵이면 64배가 된 길이의 벡터가 만들어지고, 이것을 Dense 층에 연결하면 파라미터 수가 폭발한다. 더 심각한 문제는 **입력 이미지 크기가 달라지면 모델 구조 자체를 바꿔야 한다는 것**이다.

GAP는 입력 크기에 무관하게 항상 "채널 수"만큼의 벡터를 출력한다. 8×8이든 16×16이든 채널이 128개라면 출력은 항상 (128,)이다. 덕분에 다양한 입력 크기를 하나의 모델로 처리할 수 있다.

또한 GAP는 **공간 정보를 전역 통계로 압축**하는 역할을 해서, 모델이 이미지의 특정 위치보다 전반적인 패턴에 집중하도록 유도한다. 이 때문에 CAM(Class Activation Map)처럼 모델 해석 기법에서도 GAP가 핵심 구조로 사용된다.

```
Flatten vs GAP 비교:

특징맵 (8, 8, 128)
  ↓ Flatten              ↓ GAP
  (8192,)                (128,)
  Dense(256)             Dense(256)
  파라미터: 2,097,408개   파라미터: 32,896개
  입력 크기 고정 필요     입력 크기 유연하게 사용 가능
```

```python
# GAP는 보통 마지막 Conv 블록 바로 다음에 위치
x = layers.Conv2D(128, (3,3), activation='relu', padding='same')(x)
x = layers.GlobalAveragePooling2D()(x)   # (H, W, 128) → (128,)
x = layers.Dense(256, activation='relu')(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(num_classes, activation='softmax')(x)
```

마지막 구조에서 GAP 다음에 Dense → Flatten이 오는 것이 아니라, GAP가 Flatten을 **대체**하는 구조다. GAP를 쓰면 Flatten은 필요 없어진다.

---

## 8. CAM과 Grad-CAM — 모델 해석

모델이 예측을 내렸다면, 다음 질문이 자연스럽게 따라온다. "모델이 이미지의 어느 부분을 보고 이 결론을 내렸는가?" CNN의 내부 특징맵을 이용하면 이 질문에 시각적으로 답할 수 있다.

### CAM (Class Activation Map)

GAP 구조에서만 사용 가능하다. 마지막 합성곱 층의 특징맵에 분류 가중치를 곱해서 클래스에 중요한 영역을 히트맵으로 표시한다.

```
숫자 분류 모델 적용 예시:
  원본:     3  (선명한 숫자)
  CAM:     3  (빨간 부분이 모델이 집중한 영역)
  → "숫자의 이 부분을 보고 3이라고 판단했다"
```

### Grad-CAM (Gradient-weighted CAM)

CAM의 한계(GAP 구조 필요)를 극복한 방법이다. **어떤 CNN 구조에도 적용 가능**하다. 특정 층의 출력에 대한 기울기를 이용해서 클래스 활성화 맵을 생성한다.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def grad_cam(model, img_array, layer_name, class_idx):
    """Grad-CAM 구현"""
    # 특정 층의 출력과 최종 예측을 모두 반환하는 서브 모델
    grad_model = tf.keras.Model(
        inputs=model.input,
        outputs=[model.get_layer(layer_name).output, model.output]
    )

    with tf.GradientTape() as tape:
        # 특징맵과 예측 동시 계산
        conv_outputs, predictions = grad_model(img_array[np.newaxis])
        loss = predictions[:, class_idx]   # 특정 클래스의 점수

    # 손실을 특징맵에 대해 미분
    grads = tape.gradient(loss, conv_outputs)          # 기울기
    pooled_grads = tf.reduce_mean(grads, axis=(0,1,2)) # 채널별 평균 기울기

    # 특징맵에 중요도 가중치 곱하기
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)

    return heatmap.numpy()

# 시각화
def show_gradcam(model, img_array, layer_name='last_conv'):
    class_idx = np.argmax(model.predict(img_array[np.newaxis]))
    heatmap = grad_cam(model, img_array, layer_name, class_idx)

    # 히트맵을 원본 이미지 크기로 리사이즈 후 오버레이
    heatmap_resized = tf.image.resize(
        heatmap[:,:,np.newaxis], img_array.shape[:2]
    ).numpy().squeeze()

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_array)
    axes[0].set_title('원본 이미지')
    axes[1].imshow(heatmap_resized, cmap='jet')
    axes[1].set_title('Grad-CAM 히트맵')
    axes[2].imshow(img_array)
    axes[2].imshow(heatmap_resized, cmap='jet', alpha=0.4)
    axes[2].set_title('오버레이')
    plt.tight_layout()
    plt.show()
```

---

## 9. 실전 학습 — 메모리 문제와 ImageDataGenerator

모델 구조와 해석 방법을 이해했다면, 이제 실제로 이미지 데이터를 다루며 학습할 때 마주치는 현실적인 문제를 살펴볼 차례다. 이미지 데이터는 매우 크다. 64×64 RGB 이미지 10만 장을 메모리에 한 번에 올리면 `64 × 64 × 3 × 100,000 × 4바이트 ≈ 4.9GB`가 필요하다. 실제 고해상도 이미지는 더 크다.

**ImageDataGenerator**는 이미지를 디스크에서 배치 단위로 읽으면서 **실시간으로 증강(Augmentation)** 하는 도구다. 메모리 문제를 해결하고 동시에 학습 데이터를 다양하게 만든다.

### 데이터 증강 (Data Augmentation)

원본 이미지를 다양하게 변형해서 학습 데이터를 늘리는 기법이다. 동일한 이미지를 여러 각도, 크기, 밝기로 변형하면 모델이 다양한 상황에 대한 강인성을 갖는다.

```
원본 고양이 사진 1장
  → 좌우 반전      (또 다른 학습 샘플)
  → 회전 15도      (또 다른 학습 샘플)
  → 밝기 조절      (또 다른 학습 샘플)
  → 확대/축소      (또 다른 학습 샘플)
  → 수평 이동      (또 다른 학습 샘플)
```

```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# 훈련 데이터용 — 증강 적용
train_datagen = ImageDataGenerator(
    rescale=1./255,            # 픽셀값 0~255 → 0~1 정규화
    rotation_range=20,         # ±20도 회전
    width_shift_range=0.2,     # 가로 방향 ±20% 이동
    height_shift_range=0.2,    # 세로 방향 ±20% 이동
    horizontal_flip=True,      # 좌우 반전
    zoom_range=0.2,            # 확대/축소 ±20%
    brightness_range=[0.8, 1.2],  # 밝기 조절
    shear_range=0.1,           # 전단 변환
    fill_mode='nearest'        # 변환 후 빈 공간 채우는 방법
)

# 검증/테스트 데이터용 — 정규화만, 증강 없음
val_datagen = ImageDataGenerator(rescale=1./255)

# 디렉토리에서 데이터 로드 (배치 단위)
train_generator = train_datagen.flow_from_directory(
    'data/train',
    target_size=(64, 64),      # 이미지 크기 통일
    batch_size=32,
    class_mode='categorical',  # 다중 분류
    shuffle=True,
    seed=42
)

val_generator = val_datagen.flow_from_directory(
    'data/val',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical',
    shuffle=False
)

# Generator로 학습
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=50,
    validation_data=val_generator,
    validation_steps=val_generator.samples // 32,
    callbacks=[...]
)
```

### tf.data로 대체 (현재 권장 방법)

`ImageDataGenerator`는 CPU 단에서 순차적으로 처리하기 때문에 GPU가 대기하는 시간이 생길 수 있다. TensorFlow 2.x에서는 `tf.data` 파이프라인이 권장된다.

```python
import tensorflow as tf

def load_and_preprocess(image_path, label):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [64, 64])
    img = tf.cast(img, tf.float32) / 255.0
    return img, label

def augment(img, label):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_brightness(img, max_delta=0.2)
    img = tf.image.random_contrast(img, lower=0.8, upper=1.2)
    return img, label

dataset = tf.data.Dataset.from_tensor_slices((image_paths, labels))
dataset = (dataset
    .map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    .map(augment, num_parallel_calls=tf.data.AUTOTUNE)
    .shuffle(1000)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)   # GPU 학습 중 다음 배치 미리 준비
)
```

---

## 10. 대표적인 CNN 아키텍처와 전이 학습

지금까지 CNN의 원리를 밑바닥부터 살펴봤다. 그러나 실무에서는 처음부터 CNN을 설계하기보다 수십 년간 연구로 검증된 아키텍처를 사용하거나, 대규모 데이터로 사전 학습된 모델을 가져와 내 데이터에 맞게 재조정(Fine-tuning)하는 것이 더 효율적이다.

|모델|연도|특징|파라미터 수|
|---|---|---|---|
|**LeNet**|1998|최초의 CNN (손글씨 인식)|~60K|
|**AlexNet**|2012|GPU 딥러닝 시대 개막|~62M|
|**VGG16/19**|2014|단순하고 깊은 구조 (3×3만 사용)|~138M|
|**ResNet**|2015|잔차 연결로 152층 학습 가능|~25M|
|**InceptionV3**|2015|병렬 필터 구조|~24M|
|**MobileNet**|2017|모바일 최적화 (깊이별 합성곱)|~4M|
|**EfficientNet**|2019|가장 효율적인 스케일링|~5M~66M|

### Transfer Learning (전이 학습)

대용량 데이터(ImageNet 1.2M 장)로 사전 학습된 모델의 가중치를 재사용하는 방법이다.

```python
# ResNet50 사전 학습 모델 사용
base_model = tf.keras.applications.ResNet50(
    weights='imagenet',      # ImageNet 사전 학습 가중치
    include_top=False,       # 분류 헤드 제외 (특징 추출기만)
    input_shape=(224, 224, 3)
)

# 특징 추출기 동결 (사전 학습 가중치 보존)
base_model.trainable = False

# 새로운 분류 헤드 추가
inputs  = tf.keras.Input(shape=(224, 224, 3))
x       = base_model(inputs, training=False)
x       = layers.GlobalAveragePooling2D()(x)
x       = layers.Dense(256, activation='relu')(x)
x       = layers.Dropout(0.5)(x)
outputs = layers.Dense(5, activation='softmax')(x)  # 5개 클래스

model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-3),
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# 1단계: 분류 헤드만 학습 (기본 모델 동결)
model.fit(train_dataset, epochs=10, validation_data=val_dataset)

# 2단계: Fine-tuning (기본 모델 일부 해동)
base_model.trainable = True
# 상위 30개 층만 학습 가능하게
for layer in base_model.layers[:-30]:
    layer.trainable = False

# 학습률을 크게 낮춰야 함 (사전 학습 가중치 보존)
model.compile(
    optimizer=tf.keras.optimizers.Adam(1e-5),  # 학습률 100배 낮춤
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)
model.fit(train_dataset, epochs=20, validation_data=val_dataset)
```

---

## 11. 핵심 요약

```
CNN 구성 요소:

Channel       → 이미지의 색상 차원 (RGB=3, 흑백=1)
Convolution   → 필터로 지역 패턴 감지
Filter        → 패턴 감지기. 여러 개 사용 시 다양한 특징 추출
Stride        → 필터 이동 보폭 (크면 출력 크기↓)
Padding       → 경계 처리 (same: 크기 유지 / valid: 크기 축소)
Max Pooling   → 가장 강한 특징만 남기며 크기 축소
Avg Pooling   → 평균으로 크기 축소 (GAP: 분류 직전)
Flatten / GAP → 2D 특징맵 → 1D 벡터 변환 (Dense 연결 준비)

학습 단계별 역할:
  얕은 층 → 엣지, 색상 경계 (단순)
  중간 층 → 눈, 코, 귀 등 부분 객체
  깊은 층 → 전체 객체, 고수준 개념

해석:
  CAM    → GAP 구조에서만 사용 가능
  Grad-CAM → 어떤 CNN에도 적용 가능. 모델이 집중한 영역 시각화

실무 팁:
  데이터 부족 → ImageDataGenerator 또는 tf.data + 증강
  처음부터 학습 어려움 → 사전 학습 모델 (Transfer Learning) 활용
  해석 필요 → Grad-CAM으로 모델 결정 근거 시각화
```