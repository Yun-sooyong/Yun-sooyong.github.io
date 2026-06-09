---
title: 040 Naive bayes
tag:
  - 헬스케어 ai
  - ML
description: 260609 수업 내용 정리
---

# Naive Bayes 알고리즘

나이브 베이즈는 **베이즈 정리**를 분류에 적용한 확률 기반 모델이다.  
"Naive(순진한)"라는 이름이 붙은 이유는 **모든 특성(feature)이 서로 독립**이라는 단순한 가정을 하기 때문이다.  
이 가정이 현실적으로 항상 성립하지 않더라도 실제로 텍스트 분류 등에서 놀랍도록 잘 작동한다.

---

## 1. 조건부 확률과 베이즈 정리 복습

### 조건부 확률

사건 A가 발생한 상태에서 사건 B가 발생할 확률이다.

$$P(B|A) = \frac{P(A \cap B)}{P(A)} = \frac{P(A|B) \cdot P(B)}{P(A)}$$

### 베이즈 정리

$$P(H|E) = \frac{P(H) \cdot P(E|H)}{P(E)}$$

|용어|설명|
|---|---|
|**Prior** $P(H)$|사전 확률. 증거를 보기 전 가설의 믿음|
|**Likelihood** $P(E\|H)$|우도. 가설이 참일 때 이 증거가 관측될 확률|
|**Posterior** $P(H\|E)$|사후 확률. 증거를 본 후 업데이트된 믿음|
|**Normalising Constant** $P(E)$|분모. 사후 확률의 합이 1이 되도록 정규화|

---

## 2. 스팸 분류 예시 — 베이즈 이론의 직접 적용

### 문제 설정

"VIAGRA"라는 단어가 포함된 메시지가 스팸일 확률은 얼마인가?

||VIAGRA Yes|VIAGRA No|합계|사전 확률|
|---|---|---|---|---|
|**SPAM**|4/20|16/20|20|20/100|
|**HAM**|1/80|79/80|80|80/100|
|**합계**|5/100|95/100|100||

표에서 읽어낼 수 있는 값들:

- **결합확률**: $P(\text{SPAM} \cap \text{VIAGRA}) = 4/100$
- **주변확률**: $P(\text{SPAM}) = 20/100$, $P(\text{VIAGRA}) = 5/100$
- **조건부(우도)**: $P(\text{VIAGRA}|\text{SPAM}) = 4/20 = 0.2$

### 베이즈 정리 적용

$$P(\text{SPAM}|\text{VIAGRA}) = \frac{P(\text{VIAGRA}|\text{SPAM}) \cdot P(\text{SPAM})}{P(\text{VIAGRA})} = \frac{0.2 \times 0.2}{0.05} = 0.8$$

$$P(\text{HAM}|\text{VIAGRA}) = \frac{P(\text{VIAGRA}|\text{HAM}) \cdot P(\text{HAM})}{P(\text{VIAGRA})} = \frac{(1/80) \times 0.8}{0.05} = 0.2$$

"VIAGRA"라는 단어가 있으면 그 메시지가 스팸일 확률은 **80%** 다.

```python
# 표에서 직접 계산
p_spam = 20/100     # P(SPAM) 사전 확률
p_ham  = 80/100     # P(HAM)  사전 확률

p_viagra_given_spam = 4/20     # P(VIAGRA|SPAM)
p_viagra_given_ham  = 1/80     # P(VIAGRA|HAM)

p_viagra = p_viagra_given_spam * p_spam + p_viagra_given_ham * p_ham  # 전확률

p_spam_given_viagra = (p_viagra_given_spam * p_spam) / p_viagra
p_ham_given_viagra  = (p_viagra_given_ham  * p_ham)  / p_viagra

print(f'P(SPAM|VIAGRA) = {p_spam_given_viagra:.4f}')  # 0.8
print(f'P(HAM|VIAGRA)  = {p_ham_given_viagra:.4f}')   # 0.2
print(f'합계 = {p_spam_given_viagra + p_ham_given_viagra:.4f}')  # 1.0 (정규화 확인)
```

---

## 3. Naive Bayes의 핵심 — 독립 가정으로 확장

단어가 하나일 때는 위처럼 직접 계산할 수 있다. 하지만 "VIAGRA", "FREE", "WINNER", "CLICK"처럼 여러 단어가 동시에 나타나면 어떨까?

$$P(C_1 | x_1 \cap x_2 \cap x_3 \cap x_4) = \frac{P(x_1 \cap x_2 \cap x_3 \cap x_4 | C_1) \cdot P(C_1)}{P(x_1 \cap x_2 \cap x_3 \cap x_4)}$$

분자의 $P(x_1 \cap x_2 \cap x_3 \cap x_4 | C_1)$을 직접 계산하려면 4개 단어의 모든 조합에 대한 빈도를 알아야 한다. 특성이 많아질수록 필요한 데이터가 기하급수적으로 늘어난다.

### 독립 가정 (Naive 가정)

"각 특성 $x_i$는 서로 **조건부 독립**이다" 라고 가정한다.

$$P(x_1 \cap x_2 \cap x_3 \cap x_4 | C_k) \approx P(x_1|C_k) \cdot P(x_2|C_k) \cdot P(x_3|C_k) \cdot P(x_4|C_k)$$

복잡한 결합 확률을 **각 특성의 조건부 확률의 곱**으로 단순화한다.

$$P(C_k | x_1, x_2, \ldots, x_n) \propto P(C_k) \cdot \prod_{i=1}^{n} P(x_i | C_k)$$

분모 $P(x_1, \ldots, x_n)$는 모든 클래스에서 같으므로 클래스 비교 시 약분된다.

```
최종 분류 규칙:
  y_pred = argmax_k [ P(Cₖ) × ∏ P(xᵢ|Cₖ) ]
  
  즉, 이 확률이 가장 높은 클래스로 분류한다.
```

---

## 4. MultinomialNB 확장 예시 — 날씨 분류

슬라이드의 확장된 MultinomialNB 예시다.

$$P(\text{yes}|\text{sunny}\cap\text{hot}\cap\text{high}\cap\text{windy}) = \frac{P(\text{sunny}|\text{yes}) \cdot P(\text{hot}|\text{yes}) \cdot P(\text{high}|\text{yes}) \cdot P(\text{windy}|\text{yes}) \cdot P(\text{yes})}{P(\text{sunny}) \cdot P(\text{hot}) \cdot P(\text{high}) \cdot P(\text{windy})}$$

분모는 두 클래스(yes, no) 모두에 같으므로, 실제로는 분자의 비율만 비교하면 된다.

```python
import numpy as np

# 날씨 데이터 (14일치 예시)
# feature: [sunny/overcast/rain, hot/mild/cool, high/normal, windy/not]
# label:   yes=1, no=0

# 훈련 데이터에서 계산한 조건부 확률 (예시)
p_yes = 9/14    # P(yes) 사전 확률
p_no  = 5/14    # P(no)  사전 확률

# P(feature|class) — 라플라스 평활화 포함
p_sunny_yes = (2+1)/(9+3)  # Laplace smoothing
p_hot_yes   = (2+1)/(9+3)
p_high_yes  = (3+1)/(9+2)
p_windy_yes = (3+1)/(9+2)

p_sunny_no  = (3+1)/(5+3)
p_hot_no    = (2+1)/(5+3)
p_high_no   = (4+1)/(5+2)
p_windy_no  = (3+1)/(5+2)

# 로그 확률로 계산 (언더플로우 방지)
log_p_yes = (np.log(p_yes) + np.log(p_sunny_yes) + np.log(p_hot_yes)
           + np.log(p_high_yes) + np.log(p_windy_yes))

log_p_no  = (np.log(p_no) + np.log(p_sunny_no) + np.log(p_hot_no)
           + np.log(p_high_no) + np.log(p_windy_no))

print(f'log P(yes|features) = {log_p_yes:.4f}')
print(f'log P(no|features)  = {log_p_no:.4f}')
print(f'예측: {"yes" if log_p_yes > log_p_no else "no"}')
```

---

## 5. 로그 확률 변환 — 언더플로우 방지

확률의 곱은 값이 매우 작아져 컴퓨터에서 0으로 처리되는 **언더플로우** 문제가 발생한다.  
로그를 취하면 곱셈이 덧셈으로 바뀌어 수치 안정성이 높아진다.

$$\log P(C_k|x) \propto \log P(C_k) + \sum_{i=1}^{n} \log P(x_i|C_k)$$

최종적으로 로그 확률이 가장 큰 클래스를 선택한다. 로그는 단조증가함수이므로 순서가 바뀌지 않는다.

---

## 6. 라플라스 평활화 (Laplace Smoothing)

훈련 데이터에 한 번도 등장하지 않은 단어가 있으면 $P(x_i|C_k) = 0$이 되어 아무리 다른 특성이 강력해도 전체 곱이 0이 된다. 이를 **제로 확률 문제**라 한다.

**라플라스 평활화**: 모든 빈도에 $\alpha$를 더해서 0을 방지한다.

$$P(x_i | C_k) = \frac{\text{count}(x_i, C_k) + \alpha}{\text{count}(C_k) + \alpha \cdot |V|}$$

- $\alpha$: 평활화 파라미터 (기본값 1)
- $|V|$: 전체 단어 종류 수(어휘 크기)
- $\alpha=1$이면 "각 단어가 한 번씩 등장했다고 가정"

```python
# 없는 단어도 0이 되지 않음을 확인
count_word = 0     # 훈련 데이터에 한 번도 없음
count_class = 100  # 해당 클래스의 전체 단어 수
vocab_size = 500   # 어휘 크기

alpha = 1
p_smoothed = (count_word + alpha) / (count_class + alpha * vocab_size)
print(f'라플라스 평활화 확률: {p_smoothed:.6f}')  # 0이 아닌 작은 값
```

---

## 7. sklearn의 Naive Bayes 변형 5종

특성의 **데이터 타입과 분포**에 따라 다른 클래스를 사용한다.

|클래스|수학적 가정|Feature Type|주요 사용처|
|---|---|---|---|
|**GaussianNB**|가우시안 정규분포|연속형 수치 데이터|키/몸무게, 센서 값|
|**MultinomialNB**|다항 분포|이산형 정수 데이터 (빈도)|텍스트 분류 (단어 빈도)|
|**BernoulliNB**|베르누이 분포|이진 데이터 (0/1)|텍스트 분류 (단어 있음/없음)|
|**ComplementNB**|보완적 다항 분포|불균형한 이산형 데이터|클래스 불균형 텍스트|
|**CategoricalNB**|범주형 다항 분포|명목형/범주형 데이터|범주형 특성만 있는 경우|

### GaussianNB — 연속형 특성

각 클래스 내에서 특성이 **정규분포**를 따른다고 가정한다.

$$P(x_i | C_k) = \frac{1}{\sqrt{2\pi\sigma_{ki}^2}} \exp\left(-\frac{(x_i - \mu_{ki})^2}{2\sigma_{ki}^2}\right)$$

학습 과정에서 클래스별, 특성별 평균($\mu_{ki}$)과 분산($\sigma_{ki}^2$)을 저장한다.

```python
from sklearn.naive_bayes import GaussianNB
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
gnb = GaussianNB(var_smoothing=1e-9)  # 수치 안정성을 위한 분산 평활화
gnb.fit(X, y)

# 학습된 클래스별 평균과 분산
print('클래스별 평균:')
print(gnb.theta_)   # shape: (n_classes, n_features)
print('클래스별 분산:')
print(gnb.var_)     # shape: (n_classes, n_features)
print('클래스 사전 확률:')
print(gnb.class_prior_)

y_pred = gnb.predict(X)
proba  = gnb.predict_proba(X)
print(f'\n정확도: {(y_pred == y).mean():.4f}')
```

### MultinomialNB — 텍스트 분류 (단어 빈도)

각 특성이 **다항 분포**를 따른다고 가정한다. 단어의 등장 **횟수**를 특성으로 사용한다.

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split

# 텍스트 데이터 예시
texts = [
    'buy cheap viagra now free',
    'win a million dollar prize click',
    'meeting tomorrow at 3pm office',
    'project deadline is next week',
    'free money winner click here',
    'quarterly report attached review'
]
labels = [1, 1, 0, 0, 1, 0]   # 1=SPAM, 0=HAM

# 파이프라인: 텍스트 → 단어 빈도 벡터 → MultinomialNB
pipe = make_pipeline(
    CountVectorizer(),     # 단어 빈도 행렬 생성
    MultinomialNB(alpha=1.0)   # 라플라스 평활화 alpha=1
)
pipe.fit(texts, labels)

new_texts = ['win free money now', 'meeting scheduled for friday']
pred = pipe.predict(new_texts)
prob = pipe.predict_proba(new_texts)
print(f'예측: {pred}')    # [1, 0] 예상
print(f'확률:\n{prob}')
```

### BernoulliNB — 텍스트 분류 (단어 존재 여부)

단어가 문서에 **있는지 없는지(0/1)** 만 본다. 빈도는 무시한다.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

# binary=True: 빈도가 아닌 등장 여부
pipe_bernoulli = make_pipeline(
    CountVectorizer(binary=True),   # 0 또는 1로만 변환
    BernoulliNB(alpha=1.0)
)
pipe_bernoulli.fit(texts, labels)
```

**MultinomialNB vs BernoulliNB 선택**:

- 짧은 텍스트(트윗, 제목): BernoulliNB (단어 존재 여부가 더 중요)
- 긴 문서(뉴스, 리뷰): MultinomialNB (단어 빈도가 중요한 정보)

### ComplementNB — 클래스 불균형 처리

일반 MultinomialNB는 다수 클래스에 편향되는 경향이 있다.  
ComplementNB는 각 클래스에 대해 **"이 클래스가 아닌" 나머지 클래스의 분포**로 파라미터를 추정한다. 불균형 데이터에서 MultinomialNB보다 더 나은 성능을 보인다.

```python
from sklearn.naive_bayes import ComplementNB

pipe_comp = make_pipeline(
    CountVectorizer(),
    ComplementNB(alpha=1.0)
)
pipe_comp.fit(texts, labels)
```

---

## 8. 텍스트 분류 실전 파이프라인

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

# 20 뉴스그룹 데이터셋 (실제 텍스트 분류 벤치마크)
from sklearn.datasets import fetch_20newsgroups

categories = ['alt.atheism', 'soc.religion.christian', 'comp.graphics', 'sci.med']
newsgroups = fetch_20newsgroups(subset='all', categories=categories,
                                 shuffle=True, random_state=42)

X_train, X_test, y_train, y_test = train_test_split(
    newsgroups.data, newsgroups.target,
    test_size=0.2, random_state=42, stratify=newsgroups.target
)

# TF-IDF: 단어 빈도 + 역문서 빈도 (흔한 단어의 가중치 낮춤)
pipe = make_pipeline(
    TfidfVectorizer(
        stop_words='english',    # 불용어 제거
        max_features=10000,      # 상위 10000개 단어만 사용
        ngram_range=(1, 2)       # 단어 + 2-gram
    ),
    MultinomialNB(alpha=0.1)
)

# 교차검증
cv_scores = cross_val_score(pipe, newsgroups.data, newsgroups.target, cv=5)
print(f'CV 정확도: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}')

# 최종 평가
pipe.fit(X_train, y_train)
y_pred = pipe.predict(X_test)
print(classification_report(y_test, y_pred, target_names=newsgroups.target_names))

# 새 텍스트 분류
new_docs = ['God is love', 'OpenGL on GPU', 'doctors recommend aspirin']
pred_classes = pipe.predict(new_docs)
for doc, cls in zip(new_docs, pred_classes):
    print(f'"{doc}" → {newsgroups.target_names[cls]}')
```

---

## 9. Naive Bayes의 특징과 한계

### 장점

- **매우 빠름**: 학습은 각 특성의 평균/분산/빈도만 저장하면 됨. 예측은 곱셈과 덧셈만.
- **소량 데이터에서도 잘 작동**: 파라미터 수가 적어 과적합이 드묾.
- **높은 차원에서 강함**: 텍스트 특성처럼 수만 차원도 잘 처리.
- **온라인 학습 지원**: `partial_fit()`으로 데이터를 순차적으로 추가 학습 가능.
- **확률 출력**: 클래스별 확률값을 자연스럽게 제공.

### 한계

- **독립 가정이 자주 위반**: 실제로 특성 간 상관이 있어도 독립으로 가정.  
    예: "뉴욕"과 "야구"는 같이 등장할 확률이 높지만 독립으로 처리.
- **연속형 데이터의 정규분포 가정**: GaussianNB는 특성이 정규분포를 따르지 않으면 성능이 저하.
- **확률 값이 부정확할 수 있음**: Calibration 없이는 0/1에 가까운 극단값이 많이 나옴.

```python
from sklearn.calibration import CalibratedClassifierCV

# Calibration으로 보정
gnb = GaussianNB()
calibrated_gnb = CalibratedClassifierCV(gnb, method='isotonic', cv=5)
calibrated_gnb.fit(X_train, y_train)
```

---

## 10. Naive Bayes vs 다른 분류기 비교

|구분|Naive Bayes|Logistic Regression|SVM|
|---|---|---|---|
|학습 방식|생성 모델 (분포 추정)|판별 모델 (결정 경계)|판별 모델|
|속도|매우 빠름|보통|느림 (대규모)|
|소량 데이터|강함|보통|강함|
|특성 간 독립|가정함|가정 안 함|가정 안 함|
|확률 출력|자연스러움 (but 극단적)|신뢰할 수 있음|별도 처리 필요|
|텍스트 분류|매우 강함|좋음|좋음|
|해석 가능성|높음|높음|낮음|

**생성 모델 vs 판별 모델**:

- **생성 모델 (Naive Bayes)**: $P(X|C)$를 학습 → 데이터가 어떻게 생성되었는지 학습
- **판별 모델 (LR, SVM)**: $P(C|X)$를 직접 학습 → 결정 경계를 직접 학습

---

## 11. 전체 흐름 정리

```
새 데이터 x = (x₁, x₂, ..., xₙ)

모든 클래스 k에 대해:
  score(k) = log P(Cₖ) + Σᵢ log P(xᵢ|Cₖ)
  
  ← P(Cₖ): 훈련 데이터에서 클래스 k의 비율 (사전 확률)
  ← P(xᵢ|Cₖ): 클래스 k에서 특성 xᵢ의 조건부 확률
              (GaussianNB: 정규분포 PDF)
              (MultinomialNB: 단어 빈도 비율 + 라플라스)
              (BernoulliNB: 베르누이 확률)

y_pred = argmax_k score(k)   ← 가장 높은 점수의 클래스 선택

P(Cₖ|x) 확률은 softmax(score(k))로 변환
```