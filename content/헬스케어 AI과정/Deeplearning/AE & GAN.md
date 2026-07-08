---
title: 051 AE와 GAN
tag:
  - 헬스케어 ai
  - deeplearing
  - tensorflow
description: 260707 수업 내용 정리
---

# AE와 GAN

지도학습이 정답(Target)이 주어진 데이터를 통해 입력과 출력 사이의 관계를 학습하는 것이라면, 비지도학습은 정답 없이 데이터 자체의 구조와 패턴을 학습한다. 특히 딥러닝에서는 이 비지도학습의 아이디어를 확장하여 데이터를 직접 생성해내는 모델들이 등장했는데, 이를 생성 모델(Generative Model)이라 부른다. 이번 내용에서는 대표적인 생성 모델인 Autoencoder(AE), Variational Autoencoder(VAE), GAN(Generative Adversarial Network), 그리고 이들을 보완하는 Flow-based Model과 Diffusion Model의 구조와 원리를 순서대로 살펴보고, 이들을 이해하는 데 필요한 핵심 개념인 잠재변수, KL divergence, Upsampling까지 함께 정리한다.

---

## 1. 지도학습·비지도학습과 Generative Network

지도학습과 비지도학습을 구분하는 가장 근본적인 기준은 Target(정답 레이블)의 유무다. 지도학습은 입력 `x`에 대응하는 정답 `y`가 주어져 있어, 모델이 `x → y`의 관계를 직접 학습한다. 반면 비지도학습은 정답 없이 입력 `x`만 주어지며, 모델은 데이터 내부의 숨겨진 구조나 패턴을 스스로 찾아내야 한다.

비지도학습의 목적은 크게 두 가지로 나눌 수 있다. 하나는 클러스터링(clustering)처럼 유사한 데이터끼리 묶어 그룹의 구조를 파악하는 것이고, 다른 하나는 데이터로부터 의미 있는 특징(feature)을 추출하여 이후 지도학습이나 다른 작업에 활용하는 것이다. 예를 들어 PCA(주성분분석)는 대표적인 비지도학습 기법으로, 원본 데이터의 분산을 최대한 보존하면서 더 적은 차원으로 데이터를 압축한 특징을 추출한다. 이렇게 추출된 저차원 특징은 이후 분류나 회귀 같은 후속 작업(downstream task)에 입력으로 사용될 수 있다.

Generative Network(생성 모델)는 이러한 비지도학습의 연장선에 있는 모델로, 데이터의 분포 자체를 학습하여 실제와 유사한 새로운 데이터를 만들어내는 것을 목표로 한다. 이는 입력이 주어졌을 때 정답 레이블을 예측하는 판별 모델(Discriminative Model)과 대비되는 개념이다. 판별 모델이 `P(y|x)`, 즉 데이터가 주어졌을 때 클래스를 예측하는 조건부 확률을 학습하는 반면, 생성 모델은 `P(x)` 또는 `P(x|y)`, 즉 데이터 자체가 어떻게 분포하는지를 학습한다. 이 차이 때문에 생성 모델은 학습이 끝난 후 그 분포로부터 샘플링하여 새로운 데이터를 만들어낼 수 있다.

생성 모델의 초기 형태 중 하나로 RBM(Restricted Boltzmann Machine)이 있다. RBM은 가시층(visible layer)과 은닉층(hidden layer) 두 개의 층으로 구성되며, 같은 층 내부의 노드끼리는 연결되지 않고 서로 다른 층의 노드끼리만 연결되는 제한된 구조를 갖는다. 이 구조를 통해 데이터의 확률 분포를 학습하고, 학습된 분포에서 샘플링하여 새로운 데이터를 생성할 수 있다. RBM은 이후 여러 층을 쌓은 DBN(Deep Belief Network)의 기본 구성 요소로 활용되며, 오늘날의 Autoencoder나 GAN과 같은 생성 모델들이 발전해온 역사적 출발점 중 하나로 이해할 수 있다.

---

## 2. Autoencoder (AE)

Autoencoder는 입력 데이터를 압축했다가 다시 복원하는 과정을 통해 데이터의 핵심적인 특징을 학습하는 비지도학습 신경망이다. 구조는 크게 세 부분으로 나뉜다.

```
input → [encoder] → hidden (잠재 표현) → [decoder] → output
```

입력(input)은 인코더(encoder)를 거치며 점점 더 작은 차원의 은닉층(hidden)으로 압축되고, 이 압축된 표현을 디코더(decoder)가 다시 원본과 같은 차원으로 복원하여 출력(output)을 만든다. 학습의 목표는 출력이 입력과 최대한 같아지도록 만드는 것으로, 이때 사용하는 손실 함수를 복원 손실(Reconstruction Loss)이라 부르며 보통 평균제곱오차(MSE)를 사용한다.

```
ReconLoss = (1/n) Σ (x_i - x̂_i)²
```

여기서 `x_i`는 원본 입력, `x̂_i`는 디코더가 복원한 출력이다. 흥미로운 점은 Autoencoder가 정답 레이블 없이 입력 자체를 목표로 학습하기 때문에, 형태상으로는 지도학습처럼 보이지만 실제로는 비지도학습으로 분류된다는 것이다. 이렇게 입력 자체를 정답으로 사용하는 학습 방식을 자기지도학습(Self-supervised Learning)이라 부르기도 한다.

Autoencoder가 학습을 마치면, 은닉층에 남은 압축된 표현이 원본 데이터의 핵심 특징을 담고 있게 된다. 은닉층의 차원이 입력보다 작다면(병목 구조, bottleneck), 모델은 데이터를 완전히 그대로 복사할 수 없으므로 가장 중요한 정보만 압축해서 남기도록 강제된다. 이 성질 덕분에 Autoencoder는 차원 축소, 노이즈 제거(Denoising Autoencoder), 이상치 탐지(정상 데이터는 잘 복원되지만 비정상 데이터는 복원 오차가 크다는 점을 이용) 등에 활용된다.

```python
from tensorflow.keras import layers, Model

input_dim = 784       # 예: 28x28 이미지를 1차원으로 펼친 크기
encoding_dim = 32      # 압축된 잠재 표현의 차원

inputs = layers.Input(shape=(input_dim,))
encoded = layers.Dense(128, activation='relu')(inputs)
encoded = layers.Dense(encoding_dim, activation='relu')(encoded)  # 병목 구간

decoded = layers.Dense(128, activation='relu')(encoded)
decoded = layers.Dense(input_dim, activation='sigmoid')(decoded)  # 복원

autoencoder = Model(inputs, decoded)
autoencoder.compile(optimizer='adam', loss='mse')  # 복원 손실(Reconstruction Loss)
```

---

## 3. Variational Autoencoder (VAE)

일반 Autoencoder는 입력을 하나의 고정된 점(벡터)으로 압축한다. 이 방식의 문제는, 학습에 사용된 데이터가 아닌 임의의 점을 잠재 공간에서 골라 디코더에 넣었을 때 그 점이 의미 있는 출력으로 이어질 것이라는 보장이 없다는 점이다. 즉 일반 Autoencoder는 데이터를 압축하고 복원하는 데는 뛰어나지만, 새로운 데이터를 생성하는 용도로 쓰기에는 한계가 있다.

Variational Autoencoder(VAE)는 이 문제를 해결하기 위해, 입력을 하나의 점이 아니라 확률 분포(주로 정규분포)로 인코딩한다. 인코더는 입력에 대해 평균(`μ`)과 표준편차(`σ`)라는 두 개의 값을 출력하고, 실제 잠재 벡터 `z`는 이 정규분포 `N(μ, σ)`에서 샘플링하여 얻는다. 그런데 샘플링이라는 연산 자체는 미분이 불가능하기 때문에 역전파로 `μ`와 `σ`를 학습할 수 없다는 문제가 생긴다. 이를 해결하기 위해 사용하는 기법이 재매개변수화 트릭(Reparameterization Trick)이다.

```
z = μ + σ · ε,   ε ~ N(0, 1)
```

여기서 `ε`은 표준정규분포 `N(0,1)`에서 뽑은 무작위 노이즈다. 무작위성을 `ε`이라는 별도의 항으로 분리해냄으로써, `μ`와 `σ`는 더 이상 확률적 연산의 직접적인 결과가 아니라 결정론적인 함수의 출력이 되어 역전파를 통해 정상적으로 학습할 수 있게 된다.

VAE의 손실 함수는 두 가지 항의 합으로 구성된다.

```
Loss = ReconLoss + D_KL[ N(μ, σ) ‖ N(0, 1) ]
```

첫 번째 항인 ReconLoss는 일반 Autoencoder와 마찬가지로 입력과 복원된 출력 사이의 차이를 측정한다. 두 번째 항인 KL divergence는 인코더가 만들어낸 분포 `N(μ, σ)`가 표준정규분포 `N(0, 1)`과 얼마나 다른지를 측정하여, 잠재 공간이 표준정규분포에 가깝게 정돈되도록 강제하는 정규화 항 역할을 한다. 이 항이 있기 때문에 잠재 공간의 어느 지점에서 샘플링을 하더라도 디코더가 그럴듯한 출력을 만들어낼 가능성이 높아지며, 이것이 VAE가 실제로 새로운 데이터를 생성하는 데 사용될 수 있는 이유다.

```python
from tensorflow.keras import layers, Model
import tensorflow as tf

latent_dim = 2

inputs = layers.Input(shape=(784,))
h = layers.Dense(128, activation='relu')(inputs)
z_mean = layers.Dense(latent_dim)(h)      # μ
z_log_var = layers.Dense(latent_dim)(h)   # log(σ²)

def sampling(args):
    z_mean, z_log_var = args
    epsilon = tf.random.normal(shape=tf.shape(z_mean))  # ε ~ N(0,1)
    return z_mean + tf.exp(0.5 * z_log_var) * epsilon    # 재매개변수화 트릭

z = layers.Lambda(sampling)([z_mean, z_log_var])

# 디코더
decoder_h = layers.Dense(128, activation='relu')
decoder_out = layers.Dense(784, activation='sigmoid')
outputs = decoder_out(decoder_h(z))

vae = Model(inputs, outputs)

# 손실 함수: ReconLoss + KL divergence
recon_loss = tf.reduce_mean(tf.keras.losses.binary_crossentropy(inputs, outputs)) * 784
kl_loss = -0.5 * tf.reduce_mean(1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
vae.add_loss(recon_loss + kl_loss)
vae.compile(optimizer='adam')
```

여기서 잠재변수(latent variable)라는 개념을 조금 더 짚고 넘어갈 필요가 있다. 잠재변수는 데이터를 직접 관찰할 수는 없지만, 데이터가 만들어지는 과정 뒤에 숨어서 그 데이터의 특성을 결정짓는 변수를 의미한다. 이는 흔히 빙산에 비유된다. 우리가 눈으로 보는 것은 수면 위로 드러난 빙산의 일각(관찰 가능한 데이터)뿐이지만, 실제로 그 형태를 결정하는 것은 수면 아래 훨씬 거대한 몸체(잠재변수)다. VAE의 인코더는 관찰된 데이터(빙산의 일각)로부터 그 아래에 숨어있는 잠재 표현(빙산의 몸체)을 추론하는 역할을 하고, 디코더는 반대로 잠재 표현으로부터 다시 관찰 가능한 데이터를 만들어내는 역할을 한다.

---

## 4. KL Divergence

KL divergence(Kullback-Leibler Divergence)는 두 확률분포가 서로 얼마나 다른지를 측정하는 지표로, VAE의 손실 함수뿐 아니라 통계학과 정보이론 전반에서 폭넓게 사용된다. 이산 확률분포 `p`와 `q`에 대한 KL divergence는 다음과 같이 정의된다.

```
KL(p‖q) = Σ_{k=1}^{K} p_k log(p_k / q_k)
        = Σ_{k=1}^{K} p_k log p_k − Σ_{k=1}^{K} p_k log q_k
        = −H(p) + H(p, q)
```

이 식은 두 가지 방식으로 분해해서 이해할 수 있다. 첫 번째 항 `Σ p_k log p_k`는 분포 `p`의 엔트로피에 음수 부호를 붙인 것과 같아 `−H(p)`로 쓸 수 있으며, 여기서 `H(p) = −Σ p_k log p_k`는 `p` 자체가 가진 불확실성(정보량)을 의미한다. 두 번째 항 `Σ p_k log q_k`에 음수를 붙인 것은 `p`와 `q` 사이의 교차 엔트로피(Cross Entropy) `H(p, q) = −Σ p_k log q_k`다. 따라서 `KL(p‖q) = H(p, q) − H(p)`로, "실제 분포 `p`를 두고 잘못된 분포 `q`로 표현했을 때 추가로 발생하는 정보량(비효율)"으로 해석할 수 있다. `q`가 `p`와 완전히 같아지면 `H(p,q) = H(p)`가 되어 `KL(p‖q) = 0`이 되고, `q`가 `p`와 다를수록 KL divergence 값은 커진다.

간단한 수치 예시로 살펴보면, 동전 두 개의 확률분포 `p = [0.5, 0.5]`(공정한 동전)와 `q = [0.9, 0.1]`(한쪽으로 치우친 동전)이 있다고 하자.

```
KL(p‖q) = 0.5·log(0.5/0.9) + 0.5·log(0.5/0.1)
        = 0.5·log(0.556) + 0.5·log(5)
        ≈ 0.5·(−0.587) + 0.5·(1.609)
        ≈ −0.294 + 0.805 ≈ 0.511
```

`q`가 `p`에서 많이 벗어난 분포이기 때문에 KL divergence 값이 0보다 상당히 크게 나타난다. VAE에서는 `p`를 인코더가 만든 분포 `N(μ, σ)`로, `q`를 목표로 삼는 표준정규분포 `N(0, 1)`로 두어, 인코더의 출력 분포가 표준정규분포에서 크게 벗어나지 않도록 손실 함수에 이 항을 추가하는 것이다. 또한 KL divergence는 `KL(p‖q) ≠ KL(q‖p)`로 대칭적이지 않다는 점도 기억해둘 필요가 있는데, 이 때문에 엄밀한 의미의 "거리(distance)"는 아니며 "발산(divergence)"이라는 용어를 사용한다.

---

## 5. GAN (Generative Adversarial Network)

GAN(Generative Adversarial Network)은 서로 경쟁하는 두 개의 신경망을 동시에 학습시켜 실제와 구분하기 어려운 데이터를 생성하는 모델이다. 두 신경망은 각각 생성자(Generator)와 판별자(Discriminator)로, 위조지폐범과 경찰의 관계에 자주 비유된다.

```
Noise → [Generator] → Fake Image ─┐
                                   ├─→ [Discriminator] → Real? Fake?
              Real Image ─────────┘
```

생성자(Generator)는 랜덤한 노이즈 벡터를 입력받아 진짜처럼 보이는 가짜 이미지(Fake Image)를 만들어내려고 시도한다. 판별자(Discriminator)는 실제 이미지(Real Image)와 생성자가 만든 가짜 이미지를 함께 입력받아, 각각이 진짜인지 가짜인지를 구별하려고 시도한다. 생성자는 판별자를 속이는 방향으로, 판별자는 속지 않는 방향으로 서로 대립(adversarial)하며 학습이 진행되고, 이 과정이 반복될수록 생성자는 점점 더 정교한 가짜 데이터를 만들어내게 된다.

이 관계는 다음과 같은 minimax 게임으로 수식화된다.

```
min_G max_D  V(D, G) = E_x[log D(x)] + E_z[log(1 − D(G(z)))]
```

판별자 `D`는 실제 데이터 `x`에 대해서는 `D(x)`가 1(진짜)에 가깝도록, 생성자가 만든 데이터 `G(z)`에 대해서는 `D(G(z))`가 0(가짜)에 가깝도록 만들어 `V(D,G)`를 최대화하려 한다. 반대로 생성자 `G`는 `D(G(z))`가 1에 가까워지도록, 즉 판별자를 속이는 방향으로 `V(D,G)`를 최소화하려 한다. 이상적으로 학습이 수렴하면 생성자가 만든 데이터의 분포가 실제 데이터의 분포와 거의 같아져, 판별자는 진짜와 가짜를 구분할 확률이 0.5(무작위 추측 수준)에 가까워진다.

```python
from tensorflow.keras import layers, Model
import tensorflow as tf

# Generator: 노이즈 → 가짜 이미지
def build_generator(latent_dim):
    inputs = layers.Input(shape=(latent_dim,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(784, activation='sigmoid')(x)
    return Model(inputs, x, name='generator')

# Discriminator: 이미지 → 진짜(1)/가짜(0) 확률
def build_discriminator():
    inputs = layers.Input(shape=(784,))
    x = layers.Dense(128, activation='relu')(inputs)
    x = layers.Dense(1, activation='sigmoid')(x)
    return Model(inputs, x, name='discriminator')

generator = build_generator(latent_dim=100)
discriminator = build_discriminator()
discriminator.compile(optimizer='adam', loss='binary_crossentropy')

# GAN 결합 모델 (생성자 학습 시 판별자는 고정)
discriminator.trainable = False
gan_input = layers.Input(shape=(100,))
fake_image = generator(gan_input)
validity = discriminator(fake_image)
gan = Model(gan_input, validity)
gan.compile(optimizer='adam', loss='binary_crossentropy')
```

GAN이 잘 학습되면, 잠재 공간(latent space)에 담긴 벡터들은 단순한 무작위 숫자가 아니라 의미 있는 방향성을 갖게 된다. 이 성질을 잘 보여주는 사례가 DCGAN에서 관찰된 벡터 연산(Vector Arithmetic)이다. 안경을 쓴 남자의 잠재 벡터에서 안경을 쓰지 않은 남자의 잠재 벡터를 빼고, 안경을 쓰지 않은 여자의 잠재 벡터를 더하면, 그 결과 벡터를 디코딩했을 때 안경을 쓴 여자의 이미지가 만들어진다.

```
[안경 쓴 남자] − [안경 안 쓴 남자] + [안경 안 쓴 여자] ≈ [안경 쓴 여자]
```

이는 잠재 공간에서 "안경 착용"이라는 개념이 특정한 방향(벡터)으로 일관되게 인코딩되어 있음을 보여주는 예시로, 단어 임베딩에서 `king − man + woman ≈ queen`이 성립하는 것과 개념적으로 매우 유사하다. 이러한 현상은 GAN이나 VAE 같은 생성 모델이 데이터를 단순히 암기하는 것이 아니라, 데이터 뒤에 숨어있는 의미 있는 요인들을 잠재 공간에 구조적으로 학습하고 있음을 시사한다.

VAE와 GAN 외에도 생성 모델을 만드는 방식은 여러 갈래로 발전해왔다. 그중 정확한 확률 계산을 포기하지 않으면서 데이터를 생성하려는 Flow-based Model과, 최근 이미지 생성 분야에서 가장 뛰어난 성능을 보이는 Diffusion Model을 이어서 살펴본다.

---

## 6. Flow-based Model

Flow-based Model의 핵심 목표는 다루기 쉬운 분포(예: 표준정규분포)와 실제 데이터 분포 사이를 정확하게 오갈 수 있는 가역 함수를 학습하는 것이다. VAE는 로그 우도(log-likelihood)의 하한값(ELBO)만 근사적으로 계산하고, GAN은 애초에 확률분포를 명시적으로 정의하지 않는 암묵적(implicit) 모델이라는 한계가 있었다. Flow-based Model은 이 두 모델과 달리 데이터의 확률을 정확하게(exactly) 계산하고 그대로 최대화할 수 있다는 점에서 차별화된다.

잠재 변수 `z`가 표준정규분포 `N(0, I)`를 따른다고 할 때, 어떤 가역 함수 `f`를 통해 `x = f(z)`로 실제 데이터를 만들어낼 수 있다면, 반대로 `z = f⁻¹(x)`를 통해 실제 데이터를 다시 잠재 변수로 되돌릴 수도 있다. 이때 두 확률분포 사이의 관계는 확률론의 변수 변환 공식(change of variables formula)으로 계산된다.

```
p_x(x) = p_z(f⁻¹(x)) · |det(∂f⁻¹(x) / ∂x)|
log p_x(x) = log p_z(f⁻¹(x)) + log |det(∂f⁻¹(x)/∂x)|
```

여기서 `p_z`는 잠재 변수의 확률밀도(표준정규분포이므로 계산이 간단함)이고, `det(∂f⁻¹(x)/∂x)`는 변환 `f⁻¹`의 야코비안(Jacobian) 행렬식으로, 함수 `f`가 공간을 얼마나 늘리거나 줄이는지를 보정해주는 역할을 한다. 이를 구현하려면 `f`가 반드시 가역적이어야 하고, 야코비안 행렬식을 효율적으로 계산할 수 있어야 하는데(일반적인 신경망은 이 계산이 입력 차원 `d`에 대해 `O(d³)`로 매우 비쌈), RealNVP나 Glow 같은 모델은 이를 위해 커플링 레이어(Coupling Layer)라는 구조를 사용한다.

```
y_1 = x_1
y_2 = x_2 ⊙ exp(s(x_1)) + t(x_1)
```

커플링 레이어는 입력을 `x_1`, `x_2` 두 부분으로 나눈 뒤 한쪽(`x_1`)은 그대로 두고, 다른 쪽(`x_2`)만 `x_1`에 의존하는 스케일(`s`)과 시프트(`t`) 함수로 변환한다. 이렇게 하면 야코비안 행렬이 하삼각행렬(lower-triangular matrix) 형태가 되어 행렬식이 대각 원소들의 곱(`exp(s(x_1))`의 합)만으로 간단히 계산되고, 역변환(`x_2 = (y_2 - t(y_1)) / exp(s(y_1))`)도 손쉽게 이루어진다. 여러 개의 커플링 레이어를 쌓을 때마다 고정하는 부분과 변환하는 부분을 번갈아 바꿔주면 전체 벡터의 모든 차원이 서로 영향을 주고받게 되어 표현력이 커진다.

```python
import tensorflow as tf
from tensorflow.keras import layers

class CouplingLayer(layers.Layer):
    def __init__(self, dim):
        super().__init__()
        self.scale_net = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(dim // 2, activation='tanh')  # 스케일 s(x_1)
        ])
        self.shift_net = tf.keras.Sequential([
            layers.Dense(64, activation='relu'),
            layers.Dense(dim // 2)  # 시프트 t(x_1)
        ])

    def call(self, x, reverse=False):
        x1, x2 = tf.split(x, 2, axis=-1)
        s = self.scale_net(x1)
        t = self.shift_net(x1)
        if not reverse:
            y2 = x2 * tf.exp(s) + t              # 순방향: 데이터 → 잠재 변수
            log_det = tf.reduce_sum(s, axis=-1)  # 야코비안 행렬식(로그)
            return tf.concat([x1, y2], axis=-1), log_det
        else:
            y2 = (x2 - t) * tf.exp(-s)           # 역방향: 잠재 변수 → 데이터
            return tf.concat([x1, y2], axis=-1)
```

Flow-based Model은 `f`와 `f⁻¹`을 모두 명시적으로 갖고 있어 데이터를 잠재 공간으로 압축하는 것과 복원하는 것이 완벽하게 대칭적이라는 장점이 있지만, 가역성이라는 강한 구조적 제약 때문에 표현력을 키우려면 매우 많은 층을 쌓아야 해 GAN 대비 계산 비용이 커지는 경향이 있다.

---

## 7. Diffusion Model

Diffusion Model은 최근 이미지 생성 분야(예: Stable Diffusion, DALL·E 2)에서 가장 널리 쓰이는 생성 모델로, 물리학의 확산(diffusion) 현상에서 아이디어를 얻었다. 잉크 한 방울을 물에 떨어뜨리면 시간이 지날수록 점점 퍼져나가 결국 완전히 무작위로 흩어진 상태가 되는 것처럼, Diffusion Model은 원본 데이터에 아주 조금씩 노이즈를 여러 단계에 걸쳐 더해나가 결국 순수한 노이즈로 만든 뒤, 그 반대 과정(디노이징)을 신경망이 학습하도록 한다.

이 과정은 정방향(forward)과 역방향(reverse) 확산 과정 두 단계로 구성된다. 정방향 과정은 학습이 필요 없는 고정된 절차로, 원본 데이터 `x_0`에서 시작해 `T`번의 단계를 거치며 점점 더 많은 가우시안 노이즈를 더해간다.

```
q(x_t | x_{t-1}) = N(x_t ; √(1 − β_t) · x_{t-1},  β_t · I)
```

여기서 `β_t`는 `t` 단계에서 얼마나 많은 노이즈를 더할지를 결정하는 작은 값이다. 이 정방향 과정은 가우시안 분포들의 연쇄이기 때문에, 중간 단계를 하나하나 거치지 않고도 임의의 `t` 시점의 `x_t`를 원본 `x_0`로부터 한 번에 계산하는 닫힌 형태(closed-form)의 식을 유도할 수 있다.

```
x_t = √(ᾱ_t) · x_0 + √(1 − ᾱ_t) · ε,   ε ~ N(0, I)
```

여기서 `ᾱ_t = Π_{s=1}^{t} (1 − β_s)`는 1단계부터 `t`단계까지의 노이즈 유지 비율을 누적해서 곱한 값이다. 진짜 어려운 부분은 반대 방향, 즉 노이즈 낀 `x_t`로부터 한 단계 이전의 조금 덜 노이즈 낀 `x_{t-1}`을 복원하는 역방향 과정이다. 이 조건부 분포는 직접 계산할 수 없기 때문에, 신경망 `ε_θ`가 "현재 `x_t`와 시점 `t`가 주어졌을 때 원본에 더해졌던 노이즈 `ε`이 무엇이었는지"를 예측하도록 학습시킨다.

```
L = E_{x_0, ε, t} [ ‖ ε − ε_θ(x_t, t) ‖² ]
```

즉 실제로 더해졌던 노이즈 `ε`과 신경망이 예측한 노이즈 사이의 평균제곱오차를 최소화하기만 하면 된다. 이 노이즈 예측 신경망은 보통 U-Net 구조를 사용하며, 인코더의 각 단계 출력을 디코더의 대응 단계에 직접 연결(Skip Connection)해 세밀한 공간 정보를 함께 활용한다. 학습이 끝난 뒤 새로운 데이터를 생성할 때는, 완전한 노이즈 `x_T ~ N(0, I)`에서 시작해 신경망이 예측한 노이즈를 조금씩 빼내는 과정을 반복하며 `x_{T-1}, x_{T-2}, ..., x_0`을 차례로 복원한다.

```python
import tensorflow as tf

def train_step(unet, x0, t_max, optimizer):
    batch_size = tf.shape(x0)[0]
    t = tf.random.uniform((batch_size,), 0, t_max, dtype=tf.int32)  # 무작위 시점 t 선택
    epsilon = tf.random.normal(tf.shape(x0))                        # 실제 노이즈 ε
    alpha_bar_t = get_alpha_bar(t)                                  # 미리 계산해둔 ᾱ_t
    x_t = tf.sqrt(alpha_bar_t) * x0 + tf.sqrt(1 - alpha_bar_t) * epsilon  # 닫힌 형태로 x_t 생성

    with tf.GradientTape() as tape:
        epsilon_pred = unet([x_t, t])                # 신경망이 예측한 노이즈
        loss = tf.reduce_mean(tf.square(epsilon - epsilon_pred))  # 노이즈 예측 오차
    grads = tape.gradient(loss, unet.trainable_variables)
    optimizer.apply_gradients(zip(grads, unet.trainable_variables))
    return loss

def sample(unet, shape, t_max):
    x_t = tf.random.normal(shape)  # 순수 노이즈에서 시작 (x_T)
    for t in reversed(range(t_max)):
        epsilon_pred = unet([x_t, t])
        x_t = denoise_step(x_t, epsilon_pred, t)  # 한 단계씩 노이즈를 제거하며 복원
    return x_t  # 최종적으로 x_0, 즉 생성된 데이터
```

이러한 방식을 DDPM(Denoising Diffusion Probabilistic Model)이라 부르며, 이후 등장한 DDIM(Denoising Diffusion Implicit Model)은 역방향 과정을 결정론적인 경로로 재구성해 훨씬 적은 단계만으로도 비슷한 품질의 결과를 생성할 수 있게 만들었다. 또한 조건부 생성(예: 텍스트로부터 이미지 생성)에는 분류기 없는 안내(Classifier-Free Guidance) 기법을 함께 사용해, 조건이 있는 노이즈 예측과 없는 노이즈 예측의 차이를 증폭시켜 원하는 조건에 더 강하게 부합하는 결과를 만들어낸다.

Diffusion Model은 생성 과정 중에도 이미지 크기를 다시 키워야 하는 구간이 있는데, 이는 다음으로 살펴볼 Upsampling 기법과 맞닿아 있다.

---

## 8. Upsampling: Max Pooling과 Max Unpooling

생성 모델의 디코더나 Generator는 작은 크기의 잠재 벡터로부터 점점 더 큰 크기의 이미지를 만들어내야 하는데, 이렇게 특징 맵의 크기를 키우는 과정을 Upsampling이라 한다. Upsampling을 이해하기 위해서는 먼저 그 반대 과정인 Max Pooling을 되짚어볼 필요가 있다.

Max Pooling은 입력 특징 맵을 일정 크기의 구간(예: 2x2)으로 나누고, 각 구간에서 가장 큰 값만 남겨 특징 맵의 크기를 줄이는 연산이다. 아래 예시는 4x4 입력을 2x2 구간별로 나누어 Max Pooling을 적용한 결과다.

```
Input (4x4)          Max Pooling         Output (2x2)
1 2 6 3                                  5 6
3 5 2 1        →   각 2x2 구간에서   →   7 8
7 3 4 8            최댓값만 선택
```

이때 단순히 최댓값만 남기고 그 값이 원래 어느 위치에 있었는지는 버리게 되면, 나중에 다시 크기를 복원할 때 원래의 공간적 위치 정보를 잃어버리게 된다. Max Unpooling은 이 문제를 해결하기 위해, Pooling을 수행할 당시 최댓값이 있던 위치 정보(인덱스)를 별도로 기억해두었다가, 복원 시 그 위치에만 원래 값을 되돌려놓고 나머지는 0으로 채우는 방식이다.

```
Input (2x2)         Max Unpooling            Output (4x4)
1 2                 Pooling 당시 최댓값        0 0 2 0
3 4        →        위치 정보를 사용     →     0 1 0 0
                                                0 0 0 0
                                                3 0 0 4
```

Max Unpooling은 원래 값이 있던 위치의 공간 정보를 정확히 보존한다는 장점이 있지만, Pooling 시점의 위치 인덱스를 저장해두어야 하므로 인코더와 디코더가 쌍을 이루어 위치 정보를 주고받아야 한다는 제약이 있다(대표적으로 SegNet과 같은 이미지 분할 모델에서 사용된다). 반면 GAN이나 일반적인 VAE의 디코더에서는 위치 정보를 저장해둘 대응하는 인코더가 없는 경우가 많기 때문에, Max Unpooling 대신 Transposed Convolution(전치 합성곱, Deconvolution이라고도 불림)이나 단순히 이미지를 확대한 뒤 합성곱을 적용하는 방식을 더 널리 사용한다. Transposed Convolution은 학습 가능한 필터를 사용해 작은 특징 맵의 각 값 주변에 필터를 곱해 펼쳐놓은 뒤 겹치는 부분을 더하는 방식으로 크기를 확장하며, Max Unpooling과 달리 별도의 위치 인덱스 없이도 학습을 통해 어떻게 확장할지를 스스로 익힌다는 차이가 있다.

---

## 9. 생성 모델 비교: AE vs VAE vs GAN vs Flow-based vs Diffusion

다섯 가지 생성 모델은 목적과 학습 방식, 그리고 우도 계산 가능 여부에서 뚜렷한 차이를 갖는다.

|구분|AE|VAE|GAN|Flow-based|Diffusion|
|---|---|---|---|---|---|
|주요 목적|데이터 압축·복원, 차원 축소|확률적 잠재 공간을 통한 데이터 생성|실제와 구분 어려운 데이터 생성|정확한 우도 계산과 생성|최고 품질의 반복적 데이터 생성|
|잠재 공간|고정된 점(벡터)으로 인코딩|확률분포(평균·표준편차)로 인코딩|명시적 인코더 없이 노이즈에서 직접 생성|가역 함수로 데이터와 완전히 대응|노이즈가 점진적으로 제거되는 과정 자체|
|우도 계산|해당 없음|근사값만 계산(ELBO)|불가능(암묵적 모델)|정확하게 계산 가능|명시적으로 계산하지 않음|
|학습 방식|복원 오차 최소화|복원 오차 + KL divergence 최소화|생성자·판별자의 적대적(minimax) 학습|로그 우도 직접 최대화|노이즈 예측 오차(`‖ε−ε_θ‖²`) 최소화|
|생성 속도|빠름|빠름|매우 빠름|빠름|느림(수십~수천 단계 반복)|
|생성 품질|새로운 데이터 생성에는 부적합|AE보다 매끄럽지만 다소 흐릿한 경향|매우 정교하나 불안정할 수 있음|준수하나 GAN·Diffusion보다는 낮음|현재 가장 높은 수준|
|학습 안정성|안정적|비교적 안정적|불안정(mode collapse 위험)|안정적|매우 안정적|

---

## 10. 핵심 요약 (Key Summary)

- 지도학습은 Target의 유무로 비지도학습과 구분되며, 비지도학습은 클러스터링이나 특징 추출(PCA 등)을 통해 데이터의 숨은 구조를 파악한다. Generative Network는 데이터의 분포 `P(x)` 자체를 학습해 새로운 데이터를 생성한다는 점에서 `P(y|x)`를 학습하는 Discriminative Model과 구분된다.
- Autoencoder는 input을 encoder로 압축(encode)하고 decoder로 복원(decode)하며, 복원 오차(ReconLoss)를 최소화하도록 학습하는 비지도(자기지도) 신경망이다.
- VAE는 입력을 하나의 점이 아닌 정규분포 `N(μ, σ)`로 인코딩하고, 재매개변수화 트릭 `z = μ + σ·ε`을 통해 샘플링 과정을 미분 가능하게 만들며, 손실 함수는 `ReconLoss + D_KL[N(μ,σ)‖N(0,1)]`로 구성된다.
- 잠재변수(latent variable)는 관찰되지 않지만 데이터의 특성을 결정짓는 변수로, 흔히 빙산의 몸체에 비유되며 관찰 가능한 데이터는 빙산의 일각에 해당한다.
- KL divergence는 `KL(p‖q) = H(p,q) − H(p)`로 표현되는, 두 분포의 차이를 나타내는 비대칭적인 발산 지표다.
- GAN은 Generator와 Discriminator가 `min_G max_D V(D,G)` 형태의 적대적 게임을 통해 서로를 발전시키며, 학습이 잘 된 잠재 공간에서는 `king − man + woman ≈ queen`과 유사한 방식의 벡터 연산(예: 안경 쓴 남자 − 안경 안 쓴 남자 + 안경 안 쓴 여자 ≈ 안경 쓴 여자)이 성립한다.
- Flow-based Model은 가역 함수 `f`를 통해 데이터와 잠재 변수 사이를 오가며, 변수 변환 공식 `p_x(x) = p_z(f⁻¹(x))·|det(∂f⁻¹(x)/∂x)|`으로 로그 우도를 정확하게 계산·최대화하고, 커플링 레이어로 야코비안 행렬식 계산과 역변환을 효율화한다.
- Diffusion Model은 정방향 과정에서 원본에 노이즈를 점진적으로 더해 순수한 노이즈로 만들고, 역방향 과정에서 신경망(U-Net)이 각 단계의 노이즈를 예측·제거해 `L = E[‖ε−ε_θ(x_t,t)‖²]`를 최소화하며 원본을 복원한다. DDIM은 샘플링 단계를 줄이고, Classifier-Free Guidance는 조건부 생성 품질을 높인다.
- Max Pooling은 최댓값만 남기고 위치 정보를 버리는 반면, Max Unpooling은 Pooling 당시의 위치 인덱스를 사용해 크기를 복원하며, GAN이나 VAE, Diffusion Model의 디코더에서는 이 대신 학습 가능한 Transposed Convolution을 더 널리 사용한다.