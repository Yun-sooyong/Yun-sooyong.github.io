---
title: 052 RNN, LSTM, seq2seq, NMT, self attention
tag:
  - 헬스케어 ai
  - deeplearing
  - tensorflow
description: 260708 수업 내용 정리
---

# RNN에서 Self-Attention까지: 시퀀스 모델의 발전

문장, 시계열 데이터, 음성 신호처럼 순서가 중요한 데이터를 다루려면 일반적인 완전연결 신경망(FFNN)만으로는 한계가 있다. FFNN은 입력을 한 번에 통째로 받아 처리할 뿐, 이전에 무엇이 입력되었는지에 대한 정보를 기억하지 못하기 때문이다. 이 문제를 해결하기 위해 RNN이 등장했고, RNN의 한계를 보완하기 위해 LSTM이, 문장을 다른 문장으로 변환하는 문제를 위해 Seq2Seq와 NMT가, 그리고 마지막으로 순차적인 계산 구조 자체의 한계를 극복하기 위해 Self-Attention이 차례로 등장했다. 이번 내용에서는 이 발전 흐름을 순서대로 따라가며 각 구조의 핵심 아이디어를 정리한다.

---

## 1. RNN(Recurrent Neural Network)의 구조

일반적인 FFNN(Feedforward Neural Network)은 입력층에서 은닉층을 거쳐 출력층으로 정보가 한 방향으로만 흐르며, 이전 입력에 대한 정보를 전혀 기억하지 않는다. RNN은 이 구조에 내부 피드백(internal feedback)을 추가하여, 하나의 셀(cell)이 이전 시점의 은닉 상태를 다시 자기 자신의 입력으로 받아들이도록 만든 구조다. 즉 RNN cell 하나는 본질적으로 내부 피드백을 가진 feedforward network라고 볼 수 있으며, 이 피드백 덕분에 시점마다 새로운 입력이 들어와도 이전까지의 정보를 은닉 상태에 누적해서 유지할 수 있다.

RNN cell의 계산은 다음과 같은 수식으로 표현된다.

```
a_t = tanh(W_aa · a_{t-1} + W_ax · x_t + b_a)
y_t = W_ya · a_t + b_y
```

여기서 `x_t`는 시점 `t`의 입력, `a_{t-1}`은 이전 시점에서 넘어온 은닉 상태(hidden state), `a_t`는 현재 시점의 은닉 상태, `y_t`는 현재 시점의 출력이다. `W_aa`, `W_ax`, `W_ya`는 모든 시점에서 공유되는 가중치 행렬로, 이 가중치 공유(parameter sharing) 덕분에 RNN은 입력 시퀀스의 길이가 달라져도 같은 파라미터로 동작할 수 있다.

```python
import tensorflow as tf
from tensorflow.keras import layers

# 간단한 RNN cell 예시
rnn_cell = layers.SimpleRNNCell(units=128)  # 은닉 상태 차원 128

x_t = tf.random.normal((1, 28))       # 현재 시점 입력
a_prev = tf.zeros((1, 128))           # 이전 시점 은닉 상태
output, a_t = rnn_cell(x_t, [a_prev]) # output == a_t (SimpleRNNCell은 출력과 은닉 상태가 동일)
```

RNN은 입력과 출력의 시퀀스 길이 조합에 따라 다섯 가지 대표적인 구조로 나뉜다. one-to-one은 입력과 출력이 모두 하나뿐인 구조로 일반 FFNN과 다를 바 없으며(예: 이미지 분류), one-to-many는 하나의 입력에서 여러 시점의 출력을 만들어내는 구조다(예: 이미지 하나를 보고 문장을 생성하는 이미지 캡셔닝). many-to-one은 여러 시점의 입력을 받아 마지막에 하나의 출력만 내는 구조다(예: 문장을 읽고 감성을 분류). many-to-many는 입력과 출력이 모두 여러 시점으로 이루어지는데, 여기에는 다시 두 가지 방식이 있다. 입력과 출력의 길이가 같고 매 시점마다 바로 대응하는 출력이 나오는 방식(예: 품사 태깅)과, 입력을 전부 읽은 뒤에야 출력 시퀀스를 만들기 시작하는 방식(예: 기계 번역)이다. 뒤에서 다룰 NMT는 바로 이 마지막 형태에 해당한다.

---

## 2. RNN의 한계: 장기 의존성 문제

RNN은 이론적으로 아무리 긴 시퀀스라도 처리할 수 있어야 하지만, 실제로는 시퀀스가 길어질수록 앞쪽 시점의 정보가 뒤쪽 시점까지 제대로 전달되지 못하는 장기 의존성 문제(Long-Term Dependency Problem)를 겪는다. 그 원인은 역전파 과정에서 `tanh` 함수의 미분값이 반복적으로 곱해지면서 그래디언트가 지수적으로 작아지는 기울기 소실(Vanishing Gradient), 또는 반대로 가중치 값에 따라 그래디언트가 지수적으로 커지는 기울기 폭발(Exploding Gradient)에 있다. 이 때문에 일반 RNN은 몇 십 단계 이전의 정보조차 제대로 기억하지 못하는 경우가 많았고, 이를 해결하기 위해 등장한 것이 LSTM이다.

---

## 3. LSTM (Long Short-Term Memory)

LSTM은 은닉 상태 `h_t` 외에 셀 상태(Cell State) `C_t`라는 별도의 정보 저장 경로를 추가하고, 이 셀 상태에 정보를 얼마나 지우고, 더하고, 내보낼지를 조절하는 세 개의 게이트(gate)를 도입해 장기 의존성 문제를 완화한다. 직관적으로 보면 셀 상태 `C_t`는 오래 유지되는 장기기억(long-term memory)에, 은닉 상태 `h_t`는 현재 시점에 바로 필요한 단기기억(short-term memory)에 해당한다고 볼 수 있다. 정보가 매 시점 가중치 행렬과 반복적으로 곱해지며 전달되는 일반 RNN과 달리, 장기기억에 해당하는 셀 상태는 덧셈 위주의 경로를 통해 전달되기 때문에 기울기 소실 문제가 훨씬 완화된다(다만 아주 긴 시퀀스에서는 여전히 어느 정도 발생할 수 있어, "완전히 사라진다"기보다는 "크게 억제된다"고 이해하는 것이 정확하다). 각 게이트는 0에서 1 사이의 값을 출력하는 시그모이드(`σ`) 함수로 구성되며, 이 값이 일종의 밸브처럼 정보의 흐름을 조절한다.

```
f_t = σ(W_f · [h_{t-1}, x_t] + b_f)     # Forget Gate: 이전 셀 상태를 얼마나 지울지
i_t = σ(W_i · [h_{t-1}, x_t] + b_i)     # Input Gate: 새 정보를 얼마나 반영할지
g_t = tanh(W_g · [h_{t-1}, x_t] + b_g)  # Candidate: 새로 추가할 후보 정보
o_t = σ(W_o · [h_{t-1}, x_t] + b_o)     # Output Gate: 셀 상태를 얼마나 드러낼지

C_t = f_t ⊙ C_{t-1} + i_t ⊙ g_t          # 셀 상태 업데이트
h_t = o_t ⊙ tanh(C_t)                    # 은닉 상태(출력)
```

Forget Gate(`f_t`)는 장기기억에 해당하는 이전 셀 상태 `C_{t-1}`을 얼마나 잊을지를 결정한다. Input Gate(`i_t`)는 지금 들어온 새로운 정보가 장기기억에 얼마나 반영되어야 하는지를 결정하고, 이때 실제로 추가될 후보 값은 `tanh`를 통과한 `g_t`(흔히 Candidate 또는 Gate Gate라고도 불리며, `-1`에서 `1` 사이의 값으로 새로운 정보를 표현한다)로 계산된다. Forget Gate와 Input Gate의 결과를 결합해 새로운 셀 상태 `C_t`를 만든 뒤, Output Gate(`o_t`)가 이 셀 상태 중 어느 부분을 실제 은닉 상태 `h_t`(다음 시점으로 전달되는 출력)로 내보낼지를 결정한다. 이렇게 셀 상태가 덧셈(⊙과 +) 위주로 갱신되기 때문에, 일반 RNN처럼 매 시점마다 가중치 행렬이 반복적으로 곱해지는 구조보다 그래디언트가 훨씬 안정적으로 전달된다.

```python
lstm_cell = layers.LSTMCell(units=128)

x_t = tf.random.normal((1, 28))
h_prev, c_prev = tf.zeros((1, 128)), tf.zeros((1, 128))
output, (h_t, c_t) = lstm_cell(x_t, [h_prev, c_prev])
```

---

## 4. Seq2Seq와 NMT (Neural Machine Translation)

Seq2Seq(Sequence-to-Sequence)는 입력 시퀀스 전체를 하나의 고정된 벡터(context vector)로 압축한 뒤, 이 벡터로부터 출력 시퀀스를 새로 생성하는 인코더-디코더(Encoder-Decoder) 구조다. 인코더는 RNN(또는 LSTM)으로 구성되어 입력 문장을 순서대로 읽어 들이며 마지막 시점의 은닉 상태를 context vector로 만들고, 디코더는 이 context vector를 초기 상태로 삼아 출력 문장을 한 단어씩 생성해나간다.

NMT(Neural Machine Translation, 신경망 기계 번역)는 이 Seq2Seq 구조를 기계 번역에 적용한 대표 사례다. 그런데 Seq2Seq의 기본 구조에는 근본적인 한계가 있는데, 입력 문장이 아무리 길어도 그 정보를 단 하나의 고정된 크기의 context vector에 모두 압축해야 한다는 점이다. 문장이 길어질수록 이 병목(bottleneck) 현상으로 인해 앞부분의 정보가 소실되기 쉽다. 다른 관점에서 보면 이는 "단어와 단어 사이의 관계 중요성이 고려되지 않는다"는 문제이기도 하다. 출력 문장의 특정 단어가 입력 문장의 어떤 특정 단어와 특히 밀접하게 연관되어 있더라도, Seq2Seq는 모든 입력 정보를 하나의 벡터에 동일한 방식으로 뭉뚱그려 담기 때문에 그 중요한 연관 관계를 살리지 못하고 번역 품질이 떨어지는 경우가 발생한다.

이 문제를 해결한 것이 Attention 메커니즘이다. Attention을 적용한 NMT는 디코더가 출력 단어를 하나씩 만들 때마다, 인코더가 각 시점마다 만들어낸 모든 은닉 상태를 다시 참고하되, 현재 만들고자 하는 단어와 관련이 깊은 입력 시점에 더 큰 가중치(attention weight)를 부여한다. 예를 들어 "I am a student"를 "Je suis étudiant"로 번역할 때, "étudiant"(학생)라는 단어를 생성하는 시점에서는 입력 문장 중 "student"에 해당하는 시점의 은닉 상태에 가장 큰 attention weight(예: 0.5)가 부여되고, 나머지 단어들에는 상대적으로 작은 가중치(예: 0.3, 0.1, 0.1)가 부여된다. 이렇게 가중합으로 계산된 벡터를 context vector(또는 attention vector)라 하며, 매 출력 시점마다 새로 계산된다.

```
context_vector_t = Σ_i  attention_weight_{t,i} · encoder_hidden_state_i
```

이 attention weight는 디코더의 현재 상태(추출된 특징)와 인코더의 각 은닉 상태가 얼마나 유사한지를 점수화(score)한 뒤 소프트맥스(softmax)를 취해 계산된다. 점수를 계산하는 방식에는 몇 가지가 있는데, 두 벡터를 그대로 내적(dot product)하는 방식(Luong Attention)이 가장 간단하고 직관적이며, 별도의 작은 신경망으로 두 벡터를 결합해 점수를 계산하는 방식(Bahdanau Attention, additive attention)도 널리 쓰인다. 어떤 방식이든 학습을 통해 자동으로 "어떤 입력 단어에 주목해야 하는지"를 익히게 된다는 점은 동일하다. Attention 메커니즘은 Seq2Seq의 고정 길이 병목 문제를 해결했을 뿐 아니라, 이후 등장하는 Self-Attention과 Transformer의 직접적인 토대가 되었다.

---

## 5. Self-Attention

NMT의 Attention은 디코더가 인코더를 참고하는 구조, 즉 서로 다른 두 시퀀스 사이의 관계를 다루었다. Self-Attention은 여기서 한 걸음 더 나아가, 하나의 시퀀스 내부에서 각 요소가 같은 시퀀스의 다른 모든 요소와 얼마나 관련이 있는지를 직접 계산한다. 이 계산을 위해 각 입력 벡터로부터 세 종류의 벡터, 즉 Query(질의), Key(색인), Value(값)를 만들어낸다.

Query는 "내가 지금 무엇을 찾고 있는가"를 나타내는 벡터이고, Key는 "나는 무엇을 갖고 있는가"를 나타내는 벡터이며, Value는 "실제로 전달할 내용"을 담은 벡터다. 계산 과정은 다음 세 단계로 이루어진다.

1. **Score 계산**: 하나의 입력에서 만들어진 Query 벡터를, 시퀀스 내 모든 입력의 Key 벡터와 각각 내적(dot product)하여 유사도 점수(score)를 구한다.
2. **가중치로 변환**: 이 점수들에 softmax를 적용해 합이 1이 되는 attention weight로 바꾼다.
3. **가중합**: 이 가중치를 각 입력의 Value 벡터에 곱한 뒤 모두 더해(addition), 최종 출력 벡터를 만든다.

이를 행렬로 일반화한 것이 Self-Attention의 대표적인 수식이다.

```
Attention(Q, K, V) = softmax( Q Kᵀ / √d_k ) V
```

여기서 `d_k`는 Key 벡터의 차원으로, 내적 값이 차원이 커질수록 지나치게 커지는 것을 막기 위해 `√d_k`로 나누어 스케일을 조정한다(스케일드 닷프로덕트 어텐션, Scaled Dot-Product Attention).

간단한 숫자 예시로 이 과정을 따라가보면 다음과 같다. 세 개의 입력 벡터가 있고, 각각으로부터 Query, Key, Value가 계산되었다고 하자. 첫 번째 입력의 Query와 세 입력 각각의 Key를 내적하면 세 개의 score(예: `9.8`, `1.4`, `9.1`)가 나온다. 이 score들에 softmax를 적용해 attention weight로 바꾼 뒤(가장 유사한 Key에 가장 큰 가중치가 부여됨), 각 Value 벡터에 그 가중치를 곱해서(multiplication) 모두 더하면(addition), 첫 번째 입력에 대한 Self-Attention 출력(output #1)이 완성된다. 같은 과정을 두 번째, 세 번째 입력에 대해서도 각각 반복하면 시퀀스 전체에 대한 Self-Attention 출력을 얻는다.

```python
import numpy as np

def self_attention(Q, K, V):
    d_k = K.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)              # Query · Key (스케일 조정 포함)
    weights = np.exp(scores) / np.exp(scores).sum(axis=-1, keepdims=True)  # softmax
    output = weights @ V                          # 가중치 · Value → 가중합
    return output, weights

Q = np.array([[1, 0, 1]])       # 첫 번째 입력의 query
K = np.array([[0, 1, 1], [4, 4, 0], [2, 3, 1]])  # 세 입력의 key
V = np.array([[1, 2, 3], [2, 8, 0], [2, 6, 3]])  # 세 입력의 value

output, weights = self_attention(Q, K, V)
```

Self-Attention의 가장 큰 장점은 RNN이나 LSTM처럼 이전 시점의 계산이 끝나야 다음 시점을 계산할 수 있는 순차적 구조가 아니라는 점이다. 모든 위치의 Query, Key, Value 계산과 가중합이 동시에(병렬로) 이루어질 수 있어, 학습 속도가 훨씬 빠르고 시퀀스가 길어져도 임의의 두 위치 사이의 관계를 거리와 무관하게 한 번의 계산으로 직접 반영할 수 있다.

---

## 6. RNN 구현 시 텐서 형태: 배치와 시퀀스 차원

실제로 RNN 계열 모델을 구현할 때는 입력 데이터가 배치(batch), 시퀀스 길이(timestep), 특징(feature) 세 가지 차원으로 구성된다는 점을 이해하는 것이 중요하다. 예를 들어 28x28 크기의 이미지를 한 번에 통째로 넣지 않고, 이미지를 28개의 행(row)으로 이루어진 시퀀스로 간주해 RNN에 입력하는 경우를 생각해보자. 이때 이미지 한 장은 시퀀스 길이가 28(위에서 아래로 28개의 행), 각 시점의 입력 차원이 28(한 행에 있는 28개의 픽셀 값)인 시퀀스가 된다.

```
One Vector input: 28 × 128 × 128
```

여기서 각 숫자는 순서대로 시퀀스 길이(28, 이미지의 행 수), 배치 크기(128, 한 번에 학습에 사용하는 이미지 수), 은닉 상태 차원(128, `BasicRNNCell(hidden:128)`)을 의미한다. RNN은 `x_1`부터 `x_28`까지 28개의 시점을 순서대로 입력받으며, 각 시점마다 이전 은닉 상태 `a_{t-1}`을 저장(store)해두었다가 다음 시점 계산에 사용하는 과정을 반복한다. 마지막 시점(`x_28`)까지 모두 처리한 뒤, 최종 은닉 상태를 바탕으로 최종 출력을 계산해 이미지를 분류한다.

```
Ŷ: 128 × one-hot-encoding(10)
```

이 출력은 배치 크기 128개의 이미지 각각에 대해, 10개의 클래스(예: 0~9 숫자) 중 하나를 원-핫 인코딩(one-hot encoding) 형태로 예측한 결과를 의미한다. 즉 배치 안의 이미지 128장 각각이 10차원의 확률 분포(소프트맥스 출력)로 변환되어, 가장 높은 확률을 가진 클래스가 최종 예측 클래스가 된다.

```python
import tensorflow as tf
from tensorflow.keras import layers, Model

# 28x28 이미지를 28개 시점, 각 시점 28차원 입력으로 취급 (Batch_size=128)
inputs = layers.Input(shape=(28, 28))          # (timestep=28, feature=28)
x = layers.SimpleRNN(units=128)(inputs)        # BasicRNNCell(hidden:128), 마지막 은닉 상태만 반환
outputs = layers.Dense(10, activation='softmax')(x)  # Ŷ: 10개 클래스에 대한 one-hot 형태 예측

model = Model(inputs, outputs)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# 입력 shape: (128, 28, 28) → (batch_size, timestep, feature)
```

---

## 7. Self-Attention 이후: Transformer로의 확장

Self-Attention은 그 자체로도 강력하지만, 실제 자연어 처리 모델에서는 여러 개의 Self-Attention을 동시에 사용하는 멀티헤드 어텐션(Multi-Head Attention)으로 확장되어 쓰인다. 하나의 Attention만 사용하면 한 가지 관점에서만 단어 간 관계를 포착하게 되지만, 여러 개의 Attention을 병렬로 두면 각 헤드(head)가 문법적 관계, 의미적 관계 등 서로 다른 종류의 연관성을 동시에 학습할 수 있다.

또한 Self-Attention은 입력의 순서 정보를 구조적으로 전혀 반영하지 않는다는 특징이 있다. RNN은 시점 순서대로 계산되기 때문에 순서 정보가 자연스럽게 반영되지만, Self-Attention은 모든 위치를 동시에 병렬로 계산하기 때문에 "어떤 단어가 몇 번째에 있었는지"라는 정보가 사라진다. 이를 보완하기 위해 각 입력에 위치 정보를 나타내는 포지셔널 인코딩(Positional Encoding)을 더해준다.

이렇게 Self-Attention과 Multi-Head Attention, Positional Encoding을 결합하고 RNN의 순차적 구조를 완전히 제거한 것이 Transformer 구조이며, 오늘날의 BERT, GPT 계열 모델을 포함한 대부분의 대규모 언어모델(LLM)이 이 Transformer 구조를 기반으로 한다. RNN에서 LSTM, Seq2Seq, NMT의 Attention을 거쳐 Self-Attention에 이르는 흐름은 결국 "순차적으로만 처리되던 시퀀스 모델을, 병렬로 처리 가능하면서도 장거리 의존성을 놓치지 않는 구조로 발전시켜온 과정"으로 요약할 수 있다.

---

## 8. 핵심 요약 (Key Summary)

- RNN cell은 내부 피드백을 가진 feedforward network로, `a_t = tanh(W_aa·a_{t-1} + W_ax·x_t + b_a)` 형태로 이전 은닉 상태를 반영해 시퀀스 정보를 처리한다.
- RNN의 입출력 구조는 one-to-one, one-to-many, many-to-one, many-to-many(동기형/비동기형)로 구분되며, 기계 번역은 입력을 모두 읽은 뒤 출력을 생성하는 many-to-many(비동기형) 구조에 해당한다.
- RNN은 시퀀스가 길어질수록 기울기 소실·폭발로 인한 장기 의존성 문제를 겪으며, 이를 해결하기 위해 LSTM이 등장했다.
- LSTM은 Forget Gate(`f_t`), Input Gate(`i_t`), Candidate(`g_t`), Output Gate(`o_t`)로 장기기억에 해당하는 셀 상태 `C_t`와 단기기억에 해당하는 은닉 상태 `h_t`를 조절해 정보를 선택적으로 유지·전달하며, 덧셈 위주의 갱신 구조 덕분에 기울기 소실이 크게 억제된다.
- Seq2Seq는 인코더가 입력을 하나의 context vector로 압축하고 디코더가 이를 바탕으로 출력을 생성하는 구조이며, NMT는 이를 기계 번역에 적용한 사례다. 이 구조는 단어 간 관계 중요성을 고려하지 못한다는 한계가 있으며, Attention은 이 고정 길이 병목 문제를 해결하기 위해 디코더가 매 시점마다 인코더의 모든 은닉 상태를 가중합하여 참고하도록 한다(점수 계산에는 내적 방식의 Luong Attention과 신경망 결합 방식의 Bahdanau Attention이 있다).
- Self-Attention은 하나의 시퀀스 내부에서 Query·Key·Value를 사용해 `Attention(Q,K,V) = softmax(QKᵀ/√d_k)V` 방식으로 각 요소 간의 관계를 직접 계산하며, 순차적 계산이 필요 없어 병렬 처리가 가능하다.
- RNN을 실제로 구현할 때는 입력이 (batch_size, timestep, feature) 형태의 텐서로 구성되며, 예를 들어 28x28 이미지를 28개 시점의 시퀀스로 다룰 때 배치 크기 128, 은닉 차원 128을 적용하면 최종 출력은 10개 클래스에 대한 예측이 된다.
- Self-Attention은 Multi-Head Attention과 Positional Encoding과 결합되어 Transformer 구조로 발전했으며, 이는 오늘날 대부분의 대규모 언어모델의 기반이 된다.