---
title: 048 tensorflow 2.x 구조와 keras
tag:
  - 헬스케어 ai
  - deeplearing
  - tensorflow
description: 260702 수업 내용 정리
---

# TensorFlow 2.x 구조와 Keras

TensorFlow 2.x는 Google이 만든 딥러닝 프레임워크다. 1.x 시절의 복잡한 그래프 정의 방식을 버리고 **Keras를 공식 고수준 API로 통합**했다. 덕분에 모델 설계부터 학습, 평가, 배포까지 훨씬 직관적으로 할 수 있게 됐다.

```
TensorFlow 생태계 전체 구조:

사용자 코드 (Python)
   ↓
Keras API (고수준 — 쉬운 인터페이스)
   ↓
TensorFlow 코어 (백엔드 — loss, optimizer, metrics, autograd)
   ↓
CUDA / cuDNN (GPU 병렬 연산)
   ↓
NVIDIA GPU 하드웨어
```

---

## 1. TensorFlow 2.x 전체 워크플로

### Training 단계

```
Data Design           Model Design
tf.data               Keras
TF Datasets           
      ↓                    ↓
      └──────┬─────────────┘
             ↓
   Training Distribution Strategy
   ┌─────┬──────┬──────┐
   │ CPU │ GPU  │ TPU  │  ← 어떤 하드웨어에서 학습할지 선택
   └─────┴──────┴──────┘
             ↓
        Analysis
        Tensorboard         ← 학습 과정 시각화
```

**tf.data**: 대용량 데이터를 효율적으로 파이프라인으로 처리하는 모듈. `shuffle`, `batch`, `prefetch` 등으로 GPU 대기 시간을 최소화한다.

**TF Datasets**: 학습에 바로 쓸 수 있는 공개 데이터셋 모음. MNIST, CIFAR-10, ImageNet 등이 있고 `tfds.load()`로 간단하게 불러온다.

**Distribution Strategy**: 여러 GPU 또는 여러 머신에서 학습을 분산하는 방법이다. `MirroredStrategy`는 단일 머신에서 다중 GPU를 동기화하고, `MultiWorkerMirroredStrategy`는 여러 머신에 걸쳐 분산 학습을 수행한다.

**Tensorboard**: 학습 손실, 정확도, 가중치 분포, 연산 그래프를 브라우저에서 시각적으로 확인하는 도구다.

### Deployment 단계

학습이 끝난 모델은 `SavedModel` 형식으로 저장하고, 이를 다양한 환경에 배포할 수 있다.

```
SavedModel (학습된 모델 저장)
      ↓
┌─────────────────────────────────────────┐
│  Cloud / On-prem: TensorFlow Serving    │  ← REST/gRPC API 서버
│  Android/iOS/RaspberryPi: TF Lite       │  ← 모바일/엣지 디바이스
│  Browser/Node.js: TensorFlow.js         │  ← 웹 브라우저
└─────────────────────────────────────────┘
      ↓
Model Repository: TensorFlow Hub          ← 사전 학습 모델 저장소
```

**SavedModel**: 가중치 + 연산 그래프를 함께 저장해서 언어나 플랫폼에 무관하게 불러올 수 있는 표준 형식이다.

**TensorFlow Serving**: 학습된 모델을 REST API 또는 gRPC 서버로 배포하는 도구다. 새 버전이 나와도 무중단으로 교체할 수 있다.

**TensorFlow Lite**: 모바일, 임베디드 디바이스용으로 모델을 경량화해서 배포한다. 양자화(Quantization)로 모델 크기를 줄이고 추론 속도를 높인다.

**TensorFlow Hub**: 사전 학습된(pretrained) 모델을 공유하고 재사용하는 저장소다. 이미지 임베딩, 텍스트 임베딩 등 다양한 모델을 한 줄로 불러와서 Fine-tuning에 활용한다.

```python
import tensorflow as tf
import tensorflow_hub as hub

# 모델 저장
model.save('saved_model_dir')            # SavedModel 형식 (폴더)
model.save('model.h5')                   # HDF5 형식 (파일)
model.save_weights('weights.h5')         # 가중치만 저장

# 모델 불러오기
loaded_model = tf.keras.models.load_model('saved_model_dir')

# TF Hub에서 사전 학습 모델 불러오기
embed = hub.KerasLayer(
    'https://tfhub.dev/google/nnlm-ko-dim128/2',
    input_shape=[], dtype=tf.string,
    trainable=False   # 가중치 동결. Fine-tuning 시 True로 변경
)
```

---

## 2. 다양한 딥러닝 네트워크 구조

데이터 유형과 문제에 따라 적합한 아키텍처가 다르다. TensorFlow + Keras로 이 모든 구조를 구현할 수 있다.

|네트워크|약어|적합한 데이터|대표 사용처|
|---|---|---|---|
|**완전 연결**|MLP, DNN|정형(표형) 데이터|분류, 회귀|
|**합성곱 신경망**|CNN|이미지, 영상|이미지 분류, 객체 탐지|
|**순환 신경망**|RNN, LSTM, GRU|시계열, 텍스트, 음성|번역, 예측, 감성 분석|
|**트랜스포머**|Transformer|텍스트, 이미지|GPT, BERT, ViT|
|**생성 모델**|GAN, VAE|이미지, 텍스트|이미지 생성, 데이터 증강|
|**그래프 신경망**|GNN|그래프 구조 데이터|소셜 네트워크, 분자 구조|

---

## 3. Keras란 무엇인가

Keras는 딥러닝 모델을 만들기 위한 **고수준 API(High-Level API)** 다. TensorFlow가 복잡한 수학 연산과 GPU 제어를 담당한다면, Keras는 그 위에서 사람이 이해하기 쉬운 인터페이스를 제공한다.

```
사용자
  │ layers.Dense(64, activation='relu') 한 줄
  ↓
Keras (고수준 API)
  │ 내부적으로 tf.Variable, tf.matmul, tf.nn.relu 등을 조합
  ↓
TensorFlow (저수준 API)
  │ CUDA 커널 호출, 메모리 관리
  ↓
GPU
```

Keras가 없으면 행렬 곱셈 하나도 `tf.Variable`, `tf.matmul`, `GradientTape`, `apply_gradients`를 모두 직접 써야 한다. Keras는 이 반복 작업을 `Dense`, `compile`, `fit` 세 단계로 압축한다.

### Keras의 핵심 구성요소

```
Keras
  ├─ Model       → 레이어를 묶은 학습 가능한 객체
  │    ├─ Sequential    → 순서대로 층 쌓기
  │    └─ Functional    → 자유로운 연결 구조
  │
  ├─ Layer       → 연산의 단위 (Dense, Conv2D, LSTM ...)
  │
  ├─ Optimizer   → 가중치 업데이트 방법 (Adam, SGD ...)
  │
  ├─ Loss        → 손실 함수 (MSE, CrossEntropy ...)
  │
  ├─ Metrics     → 평가 지표 (accuracy, AUC ...)
  │
  └─ Callbacks   → 학습 중 자동 실행 (EarlyStopping ...)
```

---

## 4. Keras Model — Sequential과 Functional

Keras에서 모델을 만드는 방법은 크게 두 가지다. 두 방법 모두 내부적으로 동일한 TensorFlow 연산을 수행하지만 코드 작성 방식이 다르다.

### Sequential API — 층을 순서대로 쌓기

가장 단순한 방법이다. 데이터가 입력층에서 출력층으로 한 방향으로만 흐르는 구조에 사용한다. 각 층은 이전 층의 출력을 그대로 받아서 처리한다.

Sequential을 "파이프"에 비유하면 이해하기 쉽다. 데이터가 파이프 한쪽에서 들어가서 각 처리 단계를 순서대로 거쳐 반대편으로 나온다. 중간에 분기가 없고 한 줄로 이어진다.

```python
import tensorflow as tf
from tensorflow.keras import layers

# 방법 1: 리스트로 한 번에 정의
model = tf.keras.Sequential([
    layers.Dense(128, activation='relu', input_shape=(10,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
], name='my_model')

# 방법 2: add()로 하나씩 추가 (조건부로 층을 추가할 때 유용)
model = tf.keras.Sequential(name='my_model')
model.add(layers.Dense(128, activation='relu', input_shape=(10,)))
model.add(layers.BatchNormalization())
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(1, activation='sigmoid'))

# 모델 구조 요약 출력
model.summary()
# Model: "my_model"
# ─────────────────────────────────────
# Layer (type)        Output Shape    Param #
# ─────────────────────────────────────
# dense (Dense)       (None, 128)     1,408
# batch_normalization (None, 128)     512
# dropout (Dropout)   (None, 128)     0
# dense_1 (Dense)     (None, 64)      8,256
# dense_2 (Dense)     (None, 1)       65
# ─────────────────────────────────────
# Total params: 10,241
# Trainable params: 9,985
# Non-trainable params: 256
```

### Functional API — 자유로운 연결 구조

Sequential의 한계는 입력이 하나, 출력이 하나인 직선 구조만 만들 수 있다는 것이다. Functional API는 레이어를 함수처럼 호출해서 자유롭게 연결한다. 이를 통해 다중 입출력, 잔차 연결, 분기 구조 등 복잡한 모델을 만들 수 있다.

핵심은 `tf.keras.Input()`으로 입력의 형태를 먼저 선언하고, 각 레이어를 함수처럼 호출해서 연결한 뒤, `tf.keras.Model(inputs, outputs)`으로 마무리하는 것이다.

```python
# ── 기본 Functional 모델 ──────────────────────────────
inputs = tf.keras.Input(shape=(10,), name='input')
x = layers.Dense(128, activation='relu')(inputs)
x = layers.BatchNormalization()(x)
x = layers.Dropout(0.3)(x)
x = layers.Dense(64, activation='relu')(x)
outputs = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(inputs=inputs, outputs=outputs,
                       name='functional_model')

# ── 다중 출력 모델 (분류 + 회귀를 동시에) ────────────
inputs = tf.keras.Input(shape=(10,), name='input')
shared = layers.Dense(128, activation='relu')(inputs)   # 공통 층
shared = layers.BatchNormalization()(shared)

# 분기 1: 분류 헤드
clf_branch = layers.Dense(64, activation='relu')(shared)
out_cls    = layers.Dense(1, activation='sigmoid', name='classification')(clf_branch)

# 분기 2: 회귀 헤드
reg_branch = layers.Dense(64, activation='relu')(shared)
out_reg    = layers.Dense(1, activation='linear', name='regression')(reg_branch)

multi_model = tf.keras.Model(
    inputs=inputs,
    outputs={'classification': out_cls, 'regression': out_reg}
)

# 다중 출력 모델 컴파일
multi_model.compile(
    optimizer='adam',
    loss={'classification': 'binary_crossentropy',
          'regression':     'mse'},
    metrics={'classification': 'accuracy',
             'regression':     'mae'}
)

# ── 잔차 연결 (Skip Connection) — ResNet 스타일 ───────
# 잔차 연결의 핵심: 입력을 변환한 값에 원래 입력을 더함
# → 기울기가 직접 흐르는 경로 확보 → 깊은 네트워크 학습 가능
inputs_r = tf.keras.Input(shape=(64,))
x_r      = layers.Dense(64, activation='relu')(inputs_r)
x_r      = layers.Dense(64)(x_r)              # 활성화 전 값
out_r    = layers.Add()([inputs_r, x_r])       # 원래 입력을 그대로 더함
out_r    = layers.Activation('relu')(out_r)    # 더한 후 활성화
res_model = tf.keras.Model(inputs=inputs_r, outputs=out_r)
```

### Sequential vs Functional 비교

|구분|Sequential|Functional|
|---|---|---|
|코드 단순성|매우 단순|약간 복잡|
|다중 입력|불가|가능|
|다중 출력|불가|가능|
|잔차 연결|불가|가능|
|분기 구조|불가|가능|
|내부 동작|동일|동일|
|적합한 모델|MLP, 단순 CNN|ResNet, Inception, 다중 출력|

---

## 5. Keras Model의 주요 레이어

레이어는 Keras 모델을 구성하는 기본 단위다. 각 레이어는 내부에 학습 가능한 `tf.Variable`(가중치, 편향)을 가지고 있고, `call()` 메서드에 연산이 정의되어 있다.

### Dense (완전 연결층)

가장 기본적인 레이어다. 이전 층의 모든 뉴런이 다음 층의 모든 뉴런과 연결된다.

```
연산: output = activation(input @ kernel + bias)
파라미터: kernel (가중치 행렬), bias (편향 벡터)
파라미터 수: (입력 크기 × 출력 크기) + 출력 크기
```

```python
# Dense(64): 입력 10개 → 출력 64개
layer = layers.Dense(
    units=64,                          # 출력 뉴런 수
    activation='relu',                 # 활성화 함수
    use_bias=True,                     # 편향 사용 여부
    kernel_initializer='he_normal',    # 가중치 초기화
    bias_initializer='zeros',          # 편향 초기화
    kernel_regularizer=tf.keras.regularizers.l2(0.001)  # L2 정규화
)
```

### BatchNormalization (배치 정규화)

각 층의 출력을 정규화(평균 0, 분산 1)해서 학습을 안정화한다. 학습률을 더 크게 쓸 수 있고, 초기화에 덜 민감해지고, 드롭아웃의 역할을 일부 대체한다.

```
학습 시: 배치 내 평균·분산으로 정규화 → γ, β 파라미터로 스케일·이동
추론 시: 학습 중 누적한 이동 평균·분산을 사용
```

```python
# BatchNormalization은 Dense 뒤, 활성화 함수 앞에 위치하는 것이 일반적
x = layers.Dense(128)(inputs)          # 활성화 없이
x = layers.BatchNormalization()(x)     # 정규화
x = layers.Activation('relu')(x)       # 그 다음 활성화
```

### Dropout

학습 중 무작위로 일부 뉴런을 비활성화(0으로 만들어)해서 과적합을 방지한다. 추론 시에는 비활성화하지 않고 가중치에 `(1 - rate)`를 곱해 보정한다.

```python
x = layers.Dropout(rate=0.3)(x)   # 30%의 뉴런을 무작위로 끔
# training=True/False를 Dropout이 자동으로 인식
# model.fit() → training=True (비활성화 적용)
# model.predict() → training=False (비활성화 없음)
```

### 레이어 조합 — 실전 패턴

```python
# 일반적인 은닉층 패턴:
# Dense → BatchNorm → Activation → Dropout

def dense_block(x, units, dropout_rate=0.3):
    x = layers.Dense(units)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(dropout_rate)(x)
    return x

inputs = tf.keras.Input(shape=(10,))
x = dense_block(inputs, 128)
x = dense_block(x, 64)
x = dense_block(x, 32)
outputs = layers.Dense(1, activation='sigmoid')(x)
model = tf.keras.Model(inputs, outputs)
```

---

## 6. Keras Model 5단계 라이프사이클

Keras에서 모델을 만들고 학습하는 과정이 5단계로 정해져 있다.

```
1. Define Network   → 모델 구조 설계 (Sequential / Functional)
         ↓
2. Compile Network  → 학습 설정 (loss, optimizer, metrics)
         ↓
3. Fit Network      → 학습 실행 (훈련 데이터로 반복 학습)
         ↓
4. Evaluate Network → 평가 (테스트 데이터로 성능 측정)
         ↓
5. Make Predictions → 예측 (새 데이터로 추론)
```

### Step 2 — Compile Network

학습에 필요한 세 가지 요소를 지정한다.

```python
model.compile(
    # ── Optimizer: 가중치를 어떻게 업데이트할지 ──────────
    optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.001,
        beta_1=0.9,      # 1차 모멘트 감쇠율
        beta_2=0.999     # 2차 모멘트 감쇠율
    ),

    # ── Loss: 예측과 정답의 차이를 어떻게 측정할지 ───────
    loss='binary_crossentropy',
    # loss='sparse_categorical_crossentropy'  # 다중 분류 (정수 레이블)
    # loss='categorical_crossentropy'          # 다중 분류 (one-hot)
    # loss='mse'                               # 회귀

    # ── Metrics: 학습 중 추가로 모니터링할 지표 ─────────
    metrics=['accuracy',
             tf.keras.metrics.AUC(name='auc'),
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall')]
)
```

### Step 3 — Fit Network

```python
history = model.fit(
    X_train, y_train,           # 훈련 데이터
    epochs=100,                  # 전체 데이터를 몇 번 반복 학습할지
    batch_size=32,               # 한 번에 처리할 샘플 수 (Mini-batch)
    validation_split=0.2,        # 훈련 데이터의 20%를 검증에 사용
    # validation_data=(X_val, y_val),  # 별도 검증 데이터를 지정할 때
    callbacks=[...],             # 학습 중 자동 실행할 Callback 목록
    verbose=1                    # 0: 출력 없음 / 1: 진행 바 / 2: 에포크당 한 줄
)
```

`batch_size`와 `epochs`의 관계:

```
전체 데이터 1000개, batch_size=32, epochs=10이면:

  1 에포크 = 1000/32 ≈ 32번의 가중치 업데이트
  10 에포크 = 320번의 가중치 업데이트

  각 에포크마다 데이터 순서를 섞어서 학습 (기본값: shuffle=True)
```

### Step 4 — Evaluate Network

```python
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'테스트 손실:    {test_loss:.4f}')
print(f'테스트 정확도:  {test_acc:.4f}')
```

### Step 5 — Make Predictions

```python
# 확률값 반환 (Sigmoid 출력이면 0~1)
y_pred_prob = model.predict(X_test)

# 이진 분류: 확률 → 클래스 레이블
y_pred = (y_pred_prob > 0.5).astype(int).flatten()

# 다중 분류: 가장 확률 높은 클래스
y_pred_class = y_pred_prob.argmax(axis=1)
```

---

## 7. History — 학습 이력

`model.fit()`이 반환하는 `History` 객체에 에포크별 학습 지표가 기록된다. 이것으로 과적합 여부와 수렴 상태를 진단한다.

```python
import matplotlib.pyplot as plt

history = model.fit(X_train, y_train, epochs=100,
                    validation_split=0.2, verbose=0)

# history.history는 딕셔너리
print(history.history.keys())
# dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

# 학습/검증 곡선 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['loss'],     label='훈련 손실')
axes[0].plot(history.history['val_loss'], label='검증 손실')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].set_title('손실 곡선')
axes[0].legend()

axes[1].plot(history.history['accuracy'],     label='훈련 정확도')
axes[1].plot(history.history['val_accuracy'], label='검증 정확도')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].set_title('정확도 곡선')
axes[1].legend()

plt.tight_layout()
plt.show()
```

과적합 진단 기준:

```
훈련 손실 ↓ + 검증 손실 ↑  → 과적합 (Overfitting)
  해결: Dropout 추가, L2 정규화, 데이터 증강, 모델 축소

둘 다 ↓ + 간격이 작음       → 좋은 학습 상태

둘 다 높음                   → 과소적합 (Underfitting)
  해결: 모델 복잡도 증가, 에포크 증가, 학습률 조정
```

---

## 8. Callback 객체

Callback은 학습 중 특정 시점(에포크 시작/종료, 배치 처리 후 등)에 **자동으로 실행되는 함수**다. 학습을 자동으로 제어하는 데 사용한다.

### EarlyStopping — 조기 종료

검증 손실이 더 이상 개선되지 않으면 학습을 자동으로 중단한다. 과적합을 방지하고 불필요한 학습 시간을 줄인다.

```python
early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',          # 모니터링할 지표
    patience=10,                 # 10 에포크 동안 개선 없으면 중단
    restore_best_weights=True,   # 가장 좋았던 시점의 가중치로 자동 복원
    min_delta=0.001,             # 이보다 작은 변화는 개선으로 인정하지 않음
    mode='min',                  # 'min'(손실), 'max'(정확도)
    verbose=1
)
```

### LearningRateScheduler — 학습률 동적 조절

에포크가 진행될수록 학습률을 줄여서 수렴을 안정화한다. 초반에는 큰 보폭으로 빠르게 이동하고, 후반에는 작은 보폭으로 세밀하게 조정한다.

```python
# 방법 1: 함수로 스케줄 직접 정의
def lr_schedule(epoch, lr):
    if epoch < 10:
        return lr            # 처음 10 에포크: 현재 학습률 유지
    elif epoch < 20:
        return lr * 0.5      # 10~20 에포크: 절반으로 감소
    else:
        return lr * 0.1      # 20 에포크 이후: 1/10로 감소

lr_scheduler = tf.keras.callbacks.LearningRateScheduler(
    schedule=lr_schedule, verbose=1
)

# 방법 2: ReduceLROnPlateau — 검증 손실이 정체되면 자동으로 감소
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,       # 학습률을 현재의 0.5배로 줄임
    patience=5,       # 5 에포크 정체 시 감소
    min_lr=1e-6,      # 학습률의 최소 하한
    verbose=1
)
```

### ModelCheckpoint — 최적 모델 자동 저장

학습 중 가장 성능이 좋은 모델을 자동으로 저장한다. 학습이 중간에 끊겨도 최적 모델을 보존할 수 있다.

```python
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath='best_model.h5',
    monitor='val_accuracy',
    save_best_only=True,        # 최고 성능일 때만 저장 (덮어쓰기)
    save_weights_only=False,
    mode='max',
    verbose=1
)

# 에포크별 파일명 자동 지정 (덮어쓰지 않고 모두 보관할 때)
checkpoint_all = tf.keras.callbacks.ModelCheckpoint(
    filepath='checkpoints/epoch{epoch:03d}_loss{val_loss:.4f}.h5',
    save_best_only=False,
    verbose=0
)
```

### Callback 함께 사용하기

```python
callbacks = [
    tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=15, restore_best_weights=True
    ),
    tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.5, patience=5, min_lr=1e-6
    ),
    tf.keras.callbacks.ModelCheckpoint(
        'best_model.h5', monitor='val_accuracy',
        save_best_only=True, mode='max'
    ),
    tf.keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=1)
]

history = model.fit(
    X_train, y_train,
    epochs=200,           # EarlyStopping이 알아서 멈추므로 크게 설정
    batch_size=32,
    validation_split=0.2,
    callbacks=callbacks,
    verbose=1
)

print(f'실제 학습한 에포크 수: {len(history.history["loss"])}')
```

### 커스텀 Callback

```python
class TrainingMonitor(tf.keras.callbacks.Callback):

    def on_train_begin(self, logs=None):
        print('학습 시작!')

    def on_epoch_end(self, epoch, logs=None):
        # 특정 조건에서 학습 중단
        if logs.get('val_accuracy', 0) > 0.95:
            print(f'\nEpoch {epoch}: 검증 정확도 95% 달성! 학습 중단')
            self.model.stop_training = True

    def on_train_end(self, logs=None):
        print(f'학습 완료! 최종 손실: {logs.get("loss"):.4f}')
```

---

## 9. KerasClassifier와 KerasRegressor — sklearn 연동

Keras 모델은 sklearn의 `GridSearchCV`, `cross_val_score`, `Pipeline` 등과 직접 호환되지 않는다. `KerasClassifier` / `KerasRegressor`는 Keras 모델을 sklearn 인터페이스로 감싸는 래퍼(Wrapper)다.

```
Keras 모델 (fit/predict)
   ↓ Keras wrapper (함수화)
KerasClassifier / KerasRegressor (sklearn 인터페이스)
   ↓ pipeline
sklearn Pipeline
   ↓
GridSearchCV (자동 하이퍼파라미터 탐색)
```

```python
# pip install scikeras
from scikeras.wrappers import KerasClassifier, KerasRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score
import tensorflow as tf
import numpy as np

# 모델 생성 함수 — GridSearchCV가 이 함수를 반복 호출해서 모델 생성
def create_model(n_units=64, dropout_rate=0.3, learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(n_units, activation='relu', input_shape=(10,)),
        tf.keras.layers.Dropout(dropout_rate),
        tf.keras.layers.Dense(n_units // 2, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# KerasClassifier로 래핑
keras_clf = KerasClassifier(
    model=create_model,
    epochs=50,
    batch_size=32,
    verbose=0,
    n_units=64,
    dropout_rate=0.3,
    learning_rate=0.001
)

# sklearn Pipeline에 통합
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model',  keras_clf)
])

# GridSearchCV로 하이퍼파라미터 자동 탐색
param_grid = {
    'model__n_units':       [32, 64, 128],
    'model__dropout_rate':  [0.2, 0.3, 0.5],
    'model__learning_rate': [0.001, 0.0001],
}

grid = GridSearchCV(
    pipe, param_grid, cv=5,
    scoring='accuracy', n_jobs=1, verbose=1
)

np.random.seed(42)
X = np.random.randn(200, 10).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(int)

grid.fit(X, y)
print(f'최적 파라미터: {grid.best_params_}')
print(f'최고 CV 정확도: {grid.best_score_:.4f}')
```

---

## 10. 전체 파이프라인 — 처음부터 끝까지

```python
import tensorflow as tf
from tensorflow.keras import layers, callbacks
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# 1. 데이터 준비
np.random.seed(42)
X = np.random.randn(1000, 10).astype(np.float32)
y = (X[:, 0] + X[:, 1] > 0).astype(np.float32)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test  = scaler.transform(X_test)

# 2. Define Network
model = tf.keras.Sequential([
    layers.Dense(128, activation='relu',
                 kernel_initializer='he_normal', input_shape=(10,)),
    layers.BatchNormalization(),
    layers.Dropout(0.3),
    layers.Dense(64, activation='relu', kernel_initializer='he_normal'),
    layers.Dropout(0.2),
    layers.Dense(1, activation='sigmoid')
], name='binary_clf')

model.summary()

# 3. Compile Network
model.compile(
    optimizer=tf.keras.optimizers.Adam(0.001),
    loss='binary_crossentropy',
    metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
)

# 4. Fit Network
cb_list = [
    callbacks.EarlyStopping(monitor='val_loss', patience=15,
                             restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                                 patience=5, min_lr=1e-6),
    callbacks.ModelCheckpoint('best_model.h5', monitor='val_auc',
                               save_best_only=True, mode='max'),
]

history = model.fit(
    X_train, y_train,
    epochs=200,
    batch_size=32,
    validation_split=0.2,
    callbacks=cb_list,
    verbose=1
)

# 5. Evaluate Network
results = model.evaluate(X_test, y_test, verbose=0)
for name, val in zip(model.metrics_names, results):
    print(f'{name:12s}: {val:.4f}')

# 6. Make Predictions
y_pred_prob = model.predict(X_test, verbose=0)
y_pred      = (y_pred_prob > 0.5).astype(int).flatten()

# 7. 모델 저장
model.save('final_model')           # SavedModel 형식
```

---

## 11. 전체 구조 요약

```
TensorFlow 2.x + Keras 흐름:

[데이터]
  NumPy / Pandas
     ↓ tf.data.Dataset (shuffle → batch → prefetch)

[Keras Model 정의]
  Sequential (단순한 층 쌓기)
  Functional (복잡한 연결, 다중 입출력)
     ↓
  layers.Dense → BatchNormalization → Dropout ...

[학습 설정 — Compile]
  Optimizer: Adam / SGD+Momentum / RMSprop
  Loss:      MSE(회귀) / BCE(이진) / CCE(다중분류)
  Metrics:   accuracy / AUC / Precision / Recall

[학습 실행 — Fit]
  model.fit(X, y, epochs, batch_size, callbacks)
  → History: loss / val_loss / accuracy / val_accuracy

[Callbacks]
  EarlyStopping       → 과적합 시 자동 중단
  ReduceLROnPlateau   → 정체 시 학습률 자동 감소
  ModelCheckpoint     → 최적 모델 자동 저장
  TensorBoard         → 학습 과정 시각화

[평가 / 예측]
  model.evaluate(X_test, y_test)
  model.predict(X_new)

[배포]
  SavedModel → TF Serving (API 서버)
             → TF Lite  (모바일)
             → TF.js    (웹 브라우저)

[sklearn 연동]
  KerasClassifier / KerasRegressor (wrapper)
     ↓
  Pipeline → GridSearchCV (하이퍼파라미터 탐색)
```