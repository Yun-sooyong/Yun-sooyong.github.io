---
title: 044 Text Mining
tag:
  - 헬스케어 ai
  - ML
description: 260615 수업 내용 정리
---

# 텍스트 마이닝 (Text Mining)

텍스트 마이닝은 **비정형 텍스트 데이터에서 유용한 정보와 패턴을 추출**하는 기법이다. 뉴스, 리뷰, SNS 게시글, 논문 등 인터넷에 넘쳐나는 텍스트 데이터를 분석해서 감성 분석, 토픽 모델링, 문서 분류, 검색 등에 활용한다.

컴퓨터는 텍스트를 직접 이해하지 못한다. 텍스트를 **숫자 벡터**로 변환하는 것이 텍스트 마이닝의 첫 번째 과제다.

```
텍스트 → 전처리 → 벡터 변환 → ML 모델 → 분석 결과

"I love NLP" → [0, 1, 0, 1, 0, ...] → SVM/LogisticRegression → 긍정/부정
```

---

## 1. 텍스트 표현 방법의 발전

텍스트를 숫자로 바꾸는 방법은 단순한 빈도 기반에서 의미 기반으로 발전해왔다.

```
1세대: BoW (Bag of Words)       → 단어 빈도만 봄
2세대: TF-IDF                   → 빈도 + 중요도 가중치
3세대: Word Embedding           → 단어의 의미를 벡터 공간에 표현
       (Word2Vec, FastText, GloVe, spaCy)
```

---

## 2. 전처리 파이프라인 (Preprocessing)

텍스트를 벡터로 변환하기 전에 반드시 정제 과정이 필요하다. 슬라이드의 지도학습 흐름도가 이 과정을 보여준다.

```
원시 텍스트
   ↓ Tokenization (토큰화)
   ↓ Stop-word 제거
   ↓ Stemming / Lemmatization
   ↓ POS Tagging
   ↓
정제된 텍스트 → TF-IDF 행렬 → Text Classifier → pos/neg 분류
```

### 토큰화 (Tokenization)

텍스트를 **개별 단어나 문장 단위(토큰)** 로 분리한다.

```python
text = "I love studying Natural Language Processing!"

# 단어 토큰화
tokens = text.lower().split()
# ['i', 'love', 'studying', 'natural', 'language', 'processing!']

# 구두점 제거 포함
import re
tokens = re.findall(r'\b\w+\b', text.lower())
# ['i', 'love', 'studying', 'natural', 'language', 'processing']
```

**한국어 토큰화**: 한국어는 조사, 어미 등이 붙어 있어 단순 공백 분리가 불가능하다. KoNLPy를 사용한다.

```python
from konlpy.tag import Okt

okt = Okt()
text = "자연어 처리를 공부하는 것은 재미있어요"
print(okt.morphs(text))
# ['자연어', '처리', '를', '공부', '하는', '것', '은', '재미있어요']
print(okt.nouns(text))
# ['자연어', '처리', '공부']
```

### 불용어 제거 (Stop-word Removal)

분석에 의미 없는 단어들(the, is, a, 이, 가, 을 등)을 제거한다. 이런 단어들은 빈도는 높지만 정보를 담고 있지 않아 노이즈가 된다.

```python
from nltk.corpus import stopwords
import nltk
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))
tokens = ['i', 'love', 'studying', 'natural', 'language', 'processing']
filtered = [w for w in tokens if w not in stop_words]
# ['love', 'studying', 'natural', 'language', 'processing']
# 'i'가 제거됨
```

### 어간 추출 (Stemming) & 표제어 추출 (Lemmatization)

단어의 변형을 원형으로 통합한다.

```
Stemming (어간 추출):   규칙 기반, 빠름, 때로 부정확
  running → run
  studies → studi  (정확하지 않음)

Lemmatization (표제어 추출): 사전 기반, 느리지만 정확
  running → run
  studies → study  (정확)
  better  → good   (비교급 → 원형)
```

```python
from nltk.stem import PorterStemmer, WordNetLemmatizer

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()

words = ['running', 'studies', 'better', 'goes']
for w in words:
    print(f'{w:10s} → 어간: {stemmer.stem(w):10s} 표제어: {lemmatizer.lemmatize(w, pos="v")}')
```

### POS 태깅 (Part-Of-Speech Tagging)

각 단어에 **품사(명사, 동사, 형용사 등)** 를 부착한다. 명사만 추출하거나 동사 분석에 활용한다.

```python
import nltk
nltk.download('averaged_perceptron_tagger')

tokens = ['Natural', 'language', 'processing', 'is', 'fascinating']
pos_tags = nltk.pos_tag(tokens)
# [('Natural', 'JJ'), ('language', 'NN'), ('processing', 'NN'), 
#  ('is', 'VBZ'), ('fascinating', 'JJ')]
# JJ=형용사, NN=명사, VBZ=동사

# 명사만 추출
nouns = [w for w, pos in pos_tags if pos.startswith('NN')]
```

---

## 3. BoW (Bag of Words) — 단어 주머니

### 핵심 아이디어

문서를 **단어들의 집합(주머니)** 으로 표현한다. 단어의 순서는 무시하고 **등장 횟수(빈도)만** 본다.

```
"I love NLP and NLP loves me"
  → "단어 주머니에 넣으면": {I:1, love:1, NLP:2, and:1, loves:1, me:1}
  → 순서 무시: "NLP loves me and I love NLP"도 동일한 결과
```

### Term-Document Matrix (TDM / DTM)

여러 문서를 BoW로 표현하면 행렬이 만들어진다.

슬라이드의 Term-Document Matrix 예시:

||D1|D2|D3|D4|D5|
|---|---|---|---|---|---|
|complexity|2||3|2|3|
|algorithm|3|||4|4|
|entropy|1||||2|
|traffic||2|3|||
|network||1|4|||

- **TDM (Term-Document Matrix)**: 행=단어, 열=문서
- **DTM (Document-Term Matrix)**: 행=문서, 열=단어 (TDM의 전치)

```
직관:
  complexity라는 단어가 D1에 2번, D3에 3번 나왔다
  → 두 문서는 복잡도를 다루는 주제일 가능성 높음
  → traffic, network가 없음 → 컴퓨터 과학 논문이 아닐 수 있음
```

```python
from sklearn.feature_extraction.text import CountVectorizer

corpus = [
    'I study complexity and algorithm',
    'network traffic analysis',
    'complexity entropy network traffic',
    'algorithm complexity study',
    'entropy algorithm complexity'
]

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(corpus)

print('어휘 사전:', vectorizer.vocabulary_)
print('DTM 행렬:\n', X.toarray())
print('특성 이름:', vectorizer.get_feature_names_out())
```

### BoW의 한계

```
1. 순서 무시:
   "나는 철수를 좋아해" = "철수는 나를 좋아해" → 다른 의미인데 같은 벡터

2. 의미 무시:
   "좋다"와 "훌륭하다"는 유사어인데 완전히 다른 차원으로 표현

3. 고차원 희소 벡터:
   어휘 크기 = 수만 개 → 대부분의 값이 0인 매우 큰 벡터
   → 차원의 저주, 메모리 낭비

4. 빈도 ≠ 중요도:
   "the"가 가장 많이 나온다고 가장 중요한 단어가 아님
   → TF-IDF로 해결
```

---

## 4. TF-IDF (Term Frequency - Inverse Document Frequency)

### 핵심 아이디어

단순 빈도가 아니라 **"이 단어가 이 문서에서 얼마나 중요한가"** 를 수치화한다.

```
아이디어:
  "the"는 모든 문서에 나옴 → 중요하지 않음
  "algorithm"이 특정 문서에만 나옴 → 그 문서에서 중요한 단어

TF-IDF = 이 문서에서 많이 나온다 × 다른 문서에서는 별로 안 나온다
```

### TF (Term Frequency)

문서 내에서 단어가 등장하는 빈도다.

$$TF(t, d) = \frac{\text{문서 } d \text{에서 단어 } t \text{의 등장 횟수}}{\text{문서 } d \text{의 전체 단어 수}}$$

### IDF (Inverse Document Frequency)

단어가 얼마나 **희귀한지(중요한지)** 를 나타낸다. 슬라이드의 공식:

$$IDF(t) = \log\frac{\text{총 문서 수}}{\text{단어 } t \text{가 들어 있는 문서 수}}$$

```
모든 문서에 나오는 단어:  IDF = log(N/N)  = log(1) = 0    → 중요도 0
절반 문서에 나오는 단어:  IDF = log(N/N/2) = log(2) ≈ 0.3
하나의 문서에만 나오는 단어: IDF = log(N/1) = log(N)  → 매우 높음
```

### TF-IDF 계산

$$TF\text{-}IDF(t, d) = TF(t, d) \times IDF(t)$$

```
예시 (5개 문서):
  단어 "algorithm": 문서1에 3번 등장 / 문서1은 10개 단어
                    3개 문서에 등장

  TF(algorithm, 문서1) = 3/10 = 0.3
  IDF(algorithm) = log(5/3) = log(1.67) ≈ 0.22
  TF-IDF = 0.3 × 0.22 = 0.066

  단어 "the": 5개 문서 모두에 등장
  IDF("the") = log(5/5) = log(1) = 0
  TF-IDF = 0 → 모든 문서에서 중요도 0
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

corpus = [
    'I study algorithm and complexity',
    'network traffic analysis',
    'complexity entropy network traffic analysis',
    'algorithm complexity optimization study',
    'entropy reduction algorithm complexity'
]

tfidf = TfidfVectorizer(
    max_features=20,       # 상위 20개 단어만
    stop_words='english',  # 영어 불용어 제거
    ngram_range=(1, 2),    # 1-gram과 2-gram 모두 사용
    sublinear_tf=True      # TF에 로그 스케일 적용 (log(1+TF))
)
X = tfidf.fit_transform(corpus)

feature_names = tfidf.get_feature_names_out()
print('TF-IDF 행렬 크기:', X.shape)

# 각 문서에서 가장 중요한 단어 출력
for i, doc in enumerate(corpus):
    tfidf_scores = X[i].toarray()[0]
    top_idx = np.argsort(tfidf_scores)[::-1][:3]
    print(f'문서{i+1} 핵심 단어: {[(feature_names[j], round(tfidf_scores[j], 3)) for j in top_idx]}')
```

### 코사인 유사도 — 문서 간 유사도 측정

TF-IDF 벡터 사이의 코사인 유사도로 문서 간 유사도를 측정한다. 슬라이드의 벡터 공간 그래프가 이것을 보여준다.

$$\text{cosine}(d_i, d_j) = \frac{\mathbf{d}_i \cdot \mathbf{d}_j}{|\mathbf{d}_i| |\mathbf{d}_j|}$$

```
벡터 공간에서:
  Apple = [1, 0, 0]
  Banana = [0, 1, 0]
  Orange = [0, 0, 1]

  → 세 과일이 서로 완전히 다른 방향의 벡터
  → 코사인 유사도 = 0 (완전히 다름)

쿼리 q와 가장 유사한 문서를 찾는 것 = 검색 엔진의 기본 원리
```

```python
from sklearn.metrics.pairwise import cosine_similarity

# 모든 문서 쌍의 코사인 유사도
similarity_matrix = cosine_similarity(X)

print('문서 간 유사도:')
for i in range(len(corpus)):
    for j in range(i+1, len(corpus)):
        sim = similarity_matrix[i, j]
        if sim > 0.1:
            print(f'  문서{i+1} ↔ 문서{j+1}: {sim:.3f}')

# 쿼리에 가장 유사한 문서 찾기
query = ['algorithm optimization study']
q_vec = tfidf.transform(query)
scores = cosine_similarity(q_vec, X)[0]
best_doc = scores.argmax()
print(f'\n쿼리와 가장 유사한 문서: 문서{best_doc+1} (유사도: {scores[best_doc]:.3f})')
print(f'문서 내용: {corpus[best_doc]}')
```

---

## 5. Word Embedding — 단어의 의미를 벡터로

BoW/TF-IDF의 한계는 "king"과 "queen"이 완전히 다른 벡터라는 것이다. Word Embedding은 **의미가 비슷한 단어를 비슷한 벡터 위치에 배치**한다.

```
BoW:
  king   = [0, 0, 1, 0, 0, 0, ...]  (one-hot)
  queen  = [0, 0, 0, 0, 1, 0, ...]  (완전히 다름)

Word Embedding (예: 3차원으로):
  king   = [0.9, 0.8, 0.1]
  queen  = [0.8, 0.9, 0.1]   ← 비슷한 벡터!
  man    = [0.9, 0.1, 0.9]
  woman  = [0.8, 0.1, 0.8]

king - man + woman ≈ queen  ← 단어 간 의미 관계도 연산 가능!
```

### Word2Vec (구글, 2013)

**주변 단어로 중심 단어를 예측(CBOW)** 하거나 **중심 단어로 주변 단어를 예측(Skip-gram)** 하는 신경망으로 단어 벡터를 학습한다.

```
CBOW (Continuous Bag of Words):
  입력: ["나는", "오늘", ___, "먹었다"]  (주변 단어)
  출력: "밥을"                           (중심 단어 예측)

Skip-gram:
  입력: "밥을"                           (중심 단어)
  출력: ["나는", "오늘", "먹었다"]       (주변 단어 예측)

학습 결과:
  비슷한 맥락에서 등장하는 단어 = 비슷한 벡터
  "강아지"와 "고양이"는 비슷한 맥락에서 등장 → 비슷한 벡터
```

```python
from gensim.models import Word2Vec
from konlpy.tag import Okt
import numpy as np

# 학습 데이터 준비 (문장 리스트)
sentences = [
    ['나는', '오늘', '밥을', '먹었다'],
    ['고양이', '는', '귀여운', '동물'],
    ['강아지', '는', '충성스러운', '동물'],
    ['자연어', '처리', '는', '재미있다'],
    ['머신러닝', '공부', '가', '어렵다'],
]

# Word2Vec 학습
model = Word2Vec(
    sentences=sentences,
    vector_size=100,    # 단어 벡터 차원
    window=5,           # 문맥 창 크기 (앞뒤 몇 단어까지 볼지)
    min_count=1,        # 최소 등장 횟수 (이하면 어휘에 포함 안 함)
    sg=1,               # 0=CBOW, 1=Skip-gram
    epochs=100,
    seed=42
)

# 사용
print(model.wv['고양이'])          # 단어 벡터 출력
print(model.wv.most_similar('고양이'))  # 유사 단어 탐색
print(model.wv.similarity('고양이', '강아지'))  # 유사도
```

### FastText (페이스북, 2016)

Word2Vec을 확장해서 단어를 **N-gram 조각으로 분리**해서 학습한다.

```
"playing"의 N-gram 조각 (N=3):
  "pla", "lay", "ayi", "yin", "ing"

장점: 
  학습 데이터에 없는 단어(OOV, Out-of-Vocabulary)도 처리 가능
  "playfully"가 없어도 "play", "ful", "ly" 조각으로 벡터 추정
  
  한국어처럼 조어 능력이 강한 언어에서 특히 유리
  "공부하다", "공부했다", "공부한다" → "공부" 조각 공유
```

```python
from gensim.models import FastText

ft_model = FastText(
    sentences=sentences,
    vector_size=100,
    window=5,
    min_count=1,
    min_n=2,     # N-gram 최소 길이
    max_n=6,     # N-gram 최대 길이
    epochs=100
)

# OOV 단어도 처리 가능
print(ft_model.wv['공부중'])  # 학습 데이터에 없어도 근사 벡터 반환
```

### GloVe (스탠포드, 2014) — 동시등장 통계

**전체 말뭉치에서 단어들이 함께 등장하는 통계(동시등장 행렬)** 를 기반으로 단어 벡터를 학습한다.

```
Word2Vec: 지역적 문맥 (슬라이딩 윈도우)
GloVe:    전역 통계 (전체 문서에서 단어 쌍의 등장 빈도)

"ice"와 "steam"은 반대 개념이지만 "water"와는 둘 다 자주 등장
→ 이 동시등장 비율 정보로 벡터 학습
```

```python
# GloVe는 사전 학습된 벡터를 다운로드해서 사용하는 경우가 많음
import gensim.downloader as api

glove_model = api.load('glove-wiki-gigaword-100')  # 100차원 GloVe
print(glove_model.most_similar('king'))
# [('queen', 0.75), ('prince', 0.72), ...]

# king - man + woman ≈ queen
result = glove_model.most_similar(positive=['king', 'woman'], negative=['man'])
print(result[0])  # ('queen', ...)
```

### spaCy — NLP Application 개발

spaCy는 Word Embedding 알고리즘보다는 **NLP 파이프라인 전체를 쉽게 구축**하기 위한 라이브러리다.

```
spaCy의 특징:
  - 사전 학습된 언어 모델 제공 (en_core_web_sm 등)
  - 토큰화, POS 태깅, 개체명 인식(NER), 의존성 파싱 등 통합
  - 실제 애플리케이션 개발에 최적화
  - 빠른 속도
```

```python
import spacy

nlp = spacy.load('en_core_web_sm')

doc = nlp("Apple is looking at buying UK startup for $1 billion")

for token in doc:
    print(f'{token.text:15s} {token.pos_:8s} {token.dep_:12s} {token.is_stop}')

# 개체명 인식
for ent in doc.ents:
    print(f'{ent.text} → {ent.label_}')
# Apple → ORG, UK → GPE, $1 billion → MONEY
```

### Word Embedding 방법 비교

|방법|기관|핵심 방법|특징|적합한 경우|
|---|---|---|---|---|
|**Word2Vec**|Google|CBOW / Skip-gram|지역 문맥 학습|일반적인 경우|
|**FastText**|Facebook|N-gram 조각|OOV 처리, 형태소 풍부 언어|한국어, 희귀 단어 많은 경우|
|**GloVe**|Stanford|동시등장 통계|전역 통계 반영|대량 말뭉치|
|**spaCy**|Explosion AI|여러 방법 통합|Application 개발|실제 NLP 파이프라인|

---

## 6. 텍스트 분류 — 지도학습

슬라이드 오른쪽 상단의 지도학습 흐름이다.

### 전처리 → TF-IDF → 분류기

```
원시 텍스트
   ↓ Tokenization → Stop-word 제거 → Stemming → POS 태깅
   ↓ TF-IDF 행렬 (문서 × 단어)
   ↓ Text Classifier
   ↓ pos/neg (긍정/부정)
```

### 주요 텍스트 분류 라이브러리

|라이브러리|특징|
|---|---|
|**NLTK**|자연어처리 기초 학습용. 다양한 도구 포함|
|**KoNLPy**|한국어 전용. Kkma, Komoran, Okt 등 형태소 분석기|
|**Gensim**|Word2Vec, doc2vec, LDA 등 임베딩/토픽 모델링|
|**doc2vec**|문서 전체를 하나의 벡터로 표현|
|**scikit-learn**|TF-IDF + 분류기(SVM, NB 등)|
|**RNN/LSTM**|순서 정보를 활용한 딥러닝 분류|

### 실전 텍스트 분류 파이프라인

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn.datasets import fetch_20newsgroups
import numpy as np

# 20 뉴스그룹 데이터 (실제 텍스트 분류 벤치마크)
categories = ['sci.med', 'sci.space', 'rec.sport.hockey', 'talk.politics.guns']
news = fetch_20newsgroups(subset='all', categories=categories,
                           shuffle=True, random_state=42)

X_tr, X_te, y_tr, y_te = train_test_split(
    news.data, news.target, test_size=0.2, stratify=news.target, random_state=42
)

# 다양한 분류기 비교
models = {
    'Logistic Regression': make_pipeline(
        TfidfVectorizer(max_features=10000, stop_words='english', ngram_range=(1,2)),
        LogisticRegression(max_iter=1000)
    ),
    'Multinomial NB': make_pipeline(
        TfidfVectorizer(max_features=10000, stop_words='english'),
        MultinomialNB(alpha=0.1)
    ),
    'Linear SVM': make_pipeline(
        TfidfVectorizer(max_features=10000, stop_words='english', sublinear_tf=True),
        LinearSVC(C=1.0, max_iter=5000)
    )
}

for name, model in models.items():
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    acc = (y_pred == y_te).mean()
    print(f'{name:25s}: 정확도 = {acc:.4f}')

# 가장 좋은 모델의 상세 결과
best_model = models['Linear SVM']
print('\n분류 리포트:')
print(classification_report(y_te, best_model.predict(X_te),
                             target_names=news.target_names))
```

---

## 7. LSA (Latent Semantic Analysis) — 비지도 토픽 분석

슬라이드 오른쪽 하단의 비지도학습 Topic Analysis 부분이다.

### 핵심 아이디어

TF-IDF 행렬에 **SVD(특이값 분해)** 를 적용해서 잠재된 의미(토픽)를 발견한다.

```
Term-Document Matrix (m×m):
  "고양이"와 "냥이"는 함께 등장 → 같은 의미지만 다른 열
  → SVD로 압축하면 같은 잠재 차원으로 표현

SVD 분해:
  A(m×n) = U(m×k) × Σ(k×k) × Vᵀ(k×n)
  
  U: 단어 × 잠재 토픽 행렬 (각 단어의 토픽 소속)
  Σ: 각 잠재 토픽의 중요도 (대각행렬)
  V: 문서 × 잠재 토픽 행렬 (각 문서의 토픽 소속)
```

슬라이드의 LSA 행렬 분해:

```
Term Document Matrix    Word Assignment    Topic Importance    Topic Distribution
      (m×m)          =    to Topics    ×  (n×n Diagonal)  ×   Across Documents
                          (m×n)                                    (n×m)
```

### LSA vs LDA 비교

|구분|LSA|LDA (Latent Dirichlet Allocation)|
|---|---|---|
|방법|SVD (행렬 분해)|확률 생성 모델|
|기반|선형 대수|베이즈 확률론|
|토픽 해석|어렵 (음수 포함)|쉬움 (각 토픽 = 단어 확률 분포)|
|속도|빠름|느림|
|사용|TruncatedSVD|gensim LDA|

```python
from sklearn.decomposition import TruncatedSVD, NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
import numpy as np

corpus = [
    '고양이 강아지 동물 귀여워',
    '파이썬 머신러닝 알고리즘 공부',
    '강아지 산책 동물 병원',
    '딥러닝 신경망 학습 모델',
    '고양이 귀여워 냥이 사료',
    '자연어 처리 텍스트 분류 모델'
]

# ── LSA (TF-IDF + TruncatedSVD) ────────────────────────
tfidf = TfidfVectorizer(max_features=20)
X_tfidf = tfidf.fit_transform(corpus)

lsa = TruncatedSVD(n_components=2, random_state=42)
X_lsa = lsa.fit_transform(X_tfidf)

feature_names = tfidf.get_feature_names_out()
print('LSA 토픽별 상위 단어:')
for i, comp in enumerate(lsa.components_):
    top_words = [feature_names[j] for j in comp.argsort()[:-6:-1]]
    print(f'  토픽 {i+1}: {top_words}')

# ── LDA (Count + LatentDirichletAllocation) ───────────
cv = CountVectorizer(max_features=20)
X_cv = cv.fit_transform(corpus)

lda = LatentDirichletAllocation(n_components=2, random_state=42, max_iter=100)
X_lda = lda.fit_transform(X_cv)

cv_names = cv.get_feature_names_out()
print('\nLDA 토픽별 상위 단어:')
for i, comp in enumerate(lda.components_):
    top_words = [cv_names[j] for j in comp.argsort()[:-6:-1]]
    print(f'  토픽 {i+1}: {top_words}')

# 각 문서의 토픽 소속 확률
print('\n문서별 토픽 분포:')
for i, (doc, dist) in enumerate(zip(corpus, X_lda)):
    dominant = dist.argmax() + 1
    print(f'  문서{i+1}: 토픽1={dist[0]:.3f}, 토픽2={dist[1]:.3f} → 주 토픽={dominant}')
```

---

## 8. N-gram

하나의 단어가 아닌 **연속된 N개 단어를 하나의 특성**으로 처리한다.

```
문장: "I love NLP"

1-gram (unigram):  "I", "love", "NLP"
2-gram (bigram):   "I love", "love NLP"
3-gram (trigram):  "I love NLP"

장점: 단어 순서 일부 반영
단점: 특성 수 폭발 (어휘 V개 → 최대 V² 개 bigram)
```

```python
from sklearn.feature_extraction.text import TfidfVectorizer

# N-gram 사용
tfidf_ngram = TfidfVectorizer(
    ngram_range=(1, 2),    # 1-gram + 2-gram
    max_features=5000,
    stop_words='english'
)

X_ngram = tfidf_ngram.fit_transform(corpus)
print(f'N-gram 특성 수: {X_ngram.shape[1]}')
# 1-gram만 쓸 때보다 훨씬 많은 특성
```

---

## 9. doc2vec — 문서 전체 벡터

Word2Vec이 단어 수준의 벡터를 만든다면, **doc2vec은 문서 전체를 하나의 고정 크기 벡터**로 표현한다.

```
Word2Vec: "사과" → [0.2, 0.8, 0.1, ...]
doc2vec:  "사과는 달콤한 과일입니다" → [0.3, 0.5, 0.9, ...]
```

```python
from gensim.models.doc2vec import Doc2Vec, TaggedDocument

tagged_docs = [TaggedDocument(words=doc.split(), tags=[str(i)])
               for i, doc in enumerate(corpus)]

d2v_model = Doc2Vec(
    vector_size=50,
    window=2,
    min_count=1,
    epochs=100,
    seed=42
)
d2v_model.build_vocab(tagged_docs)
d2v_model.train(tagged_docs, total_examples=d2v_model.corpus_count,
                epochs=d2v_model.epochs)

# 문서 벡터 추출
doc_vector = d2v_model.dv['0']   # 첫 번째 문서 벡터
print(f'문서 벡터 크기: {doc_vector.shape}')

# 유사 문서 찾기
similar_docs = d2v_model.dv.most_similar('0')
print(f'문서0과 유사한 문서: {similar_docs}')
```

---

## 10. 한국어 텍스트 처리 — KoNLPy

한국어는 교착어로 어미와 조사가 단어에 붙어 있어 영어와는 다른 전처리가 필요하다.

```
영어: "I study NLP" → ["I", "study", "NLP"]  (공백 분리)
한국어: "나는 자연어처리를 공부한다" → 공백 분리만으로는 불충분
        → "나/는/자연어/처리/를/공부/한/다" (형태소 분석 필요)
```

```python
from konlpy.tag import Okt, Kkma, Komoran
from collections import Counter

text = """
자연어 처리는 인공지능의 한 분야로서 컴퓨터가 인간의 언어를 이해하고
처리할 수 있도록 하는 기술입니다. 텍스트 분류, 감성 분석, 기계 번역 등
다양한 응용 분야가 있습니다.
"""

okt = Okt()

# 형태소 분석
morphs   = okt.morphs(text)
nouns    = okt.nouns(text)
pos_tags = okt.pos(text)

print(f'형태소: {morphs[:10]}')
print(f'명사:   {nouns[:10]}')

# 불용어 제거 후 명사 빈도
stopwords_ko = {'것', '수', '및', '이', '가', '를', '은', '는'}
filtered_nouns = [w for w in nouns if w not in stopwords_ko and len(w) > 1]

counter = Counter(filtered_nouns)
print(f'\n상위 10개 명사: {counter.most_common(10)}')
```

---

## 11. 워드 클라우드 (Word Cloud)

슬라이드 왼쪽 하단의 시각화다. 단어의 빈도나 중요도에 비례해 크기를 조절해서 시각화한다.

```python
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from collections import Counter

# 단어 빈도 계산
word_freq = Counter(filtered_nouns)

# 워드 클라우드 생성
wc = WordCloud(
    font_path='/usr/share/fonts/truetype/nanum/NanumGothic.ttf',  # 한글 폰트
    background_color='white',
    max_words=100,
    width=800, height=400
)

wordcloud = wc.generate_from_frequencies(word_freq)

plt.figure(figsize=(12, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis('off')
plt.title('워드 클라우드')
plt.tight_layout()
plt.show()
```

---

## 12. 텍스트 마이닝 전체 흐름 정리

```
원시 텍스트 데이터
   ↓
전처리
  토큰화 → 불용어 제거 → 어간/표제어 추출 → POS 태깅
   ↓
벡터 표현 선택
  BoW:        단순 빈도, 빠름, 의미 무시
  TF-IDF:     빈도 + 중요도 가중치, 문서 분류/검색에 강함
  Word2Vec:   단어 의미 벡터, CBOW/Skip-gram
  FastText:   N-gram 기반, OOV 처리, 한국어에 강함
  GloVe:      동시등장 통계 기반 전역 임베딩
   ↓
분석 목적에 따라 분기
  지도학습 (레이블 있음):
    감성 분석, 스팸 분류, 주제 분류
    → TF-IDF + (SVM, LogisticRegression, MultinomialNB)
    → Word Embedding + RNN/LSTM/BERT

  비지도학습 (레이블 없음):
    토픽 모델링, 문서 군집화
    → LSA (TF-IDF + TruncatedSVD)
    → LDA (CountVectorizer + LatentDirichletAllocation)
    → NMF (TF-IDF + NMF)
   ↓
결과 해석
  분류 결과, 토픽 단어 목록, 유사 문서, 워드 클라우드
```

### 라이브러리 선택 가이드

|목적|권장 라이브러리|
|---|---|
|기초 전처리 (영어)|NLTK|
|기초 전처리 (한국어)|KoNLPy (Okt, Kkma, Komoran)|
|Word Embedding 학습|Gensim (Word2Vec, FastText, doc2vec)|
|사전 학습 임베딩 활용|GloVe, BERT (transformers 라이브러리)|
|NLP 애플리케이션 개발|spaCy|
|TF-IDF + 전통 ML 분류|scikit-learn|
|토픽 모델링|Gensim (LDA), scikit-learn (LDA, NMF, LSA)|
|딥러닝 텍스트 분류|PyTorch + transformers (BERT, RoBERTa 등)|