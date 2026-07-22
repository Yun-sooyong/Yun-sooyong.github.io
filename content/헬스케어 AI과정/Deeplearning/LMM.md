---
title: 053 LMM
tag:
  - 헬스케어 ai
  - deeplearing
  - tensorflow
description: 260709 수업 내용 정리
---

# Transformer와 LLM

Transformer는 2017년 Google이 발표한 논문 "Attention Is All You Need"에서 등장한 신경망 구조다. RNN과 LSTM이 순서대로 단어를 처리하던 방식을 버리고, **모든 단어를 동시에 처리하면서 서로 얼마나 관련 있는지를 계산**하는 Attention 메커니즘을 중심으로 설계됐다. 이 구조가 현재 GPT, BERT, ChatGPT, Claude 등 거의 모든 대형 언어 모델(LLM)의 기반이 되고 있다.

---

## 1. Transformer 구조

Transformer는 두 부분으로 구성된다. 입력을 이해하는 **Encoder**와 출력을 생성하는 **Decoder**다. 이 두 파트를 어떻게 조합하느냐에 따라 모델의 특성이 달라진다.

### 전체 구조 개요

```
입력 문장                          출력 문장
"I love NLP"                      "나는 NLP를 좋아한다"

Inputs                            Outputs (shifted right)
  ↓                                     ↓
Input Embedding                   Output Embedding
  ↓                                     ↓
Positional Encoding               Positional Encoding
  ↓                                     ↓
┌──────────────────┐         ┌──────────────────────────┐
│    Encoder × N   │         │       Decoder × N         │
│                  │         │                            │
│  Multi-Head      │ ──────→ │  Masked Multi-Head        │
│  Attention       │         │  Attention                 │
│      ↓           │         │      ↓                     │
│  Add & Norm      │         │  Multi-Head Attention      │
│      ↓           │         │  (Encoder 출력 참조)       │
│  Feed Forward    │         │      ↓                     │
│      ↓           │         │  Feed Forward              │
│  Add & Norm      │         │      ↓                     │
└──────────────────┘         │  Add & Norm                │
                             └──────────────────────────┘
                                        ↓
                                      Linear
                                        ↓
                                      Softmax
                                        ↓
                               Output Probabilities
```

### Positional Encoding — 순서 정보 주입

Transformer는 모든 단어를 동시에 처리하기 때문에, 단어의 위치 정보를 따로 추가해야 한다. Positional Encoding은 각 위치마다 고유한 패턴의 숫자를 임베딩 벡터에 더해서 "이 단어가 몇 번째인지"를 모델에 알려준다.

```
Input Embedding:  [0.5, 0.2, 0.8, ...]  ← 단어의 의미
Positional Enc:   [0.0, 1.0, 0.0, ...]  ← 단어의 위치
                 +
              = [0.5, 1.2, 0.8, ...]  ← 의미 + 위치 정보
```

### Multi-Head Attention

Attention은 "이 단어를 처리할 때 어떤 다른 단어들에 집중해야 하는가"를 학습한다. Multi-Head는 이 집중 패턴을 여러 관점에서 동시에 계산한다는 의미다.

```
"The animal didn't cross the street because it was too tired"

"it"을 처리할 때:
  Head 1: "animal"에 집중 (지시 대상)
  Head 2: "tired"에 집중 (상태 설명)
  Head 3: "street"과 관계 확인 (배제)
  → 여러 Head가 서로 다른 관계를 동시에 포착
```

Attention 계산 수식:

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

- Q (Query): "나는 무엇을 찾고 있는가?"
- K (Key): "나는 무엇과 관련될 수 있는가?"
- V (Value): "실제로 가져갈 정보는 무엇인가?"
- $\sqrt{d_k}$: 차원 수의 제곱근으로 나눠서 기울기 소실 방지

### Masked Multi-Head Attention (Decoder에서만)

Decoder에서 사용하는 Attention이다. 미래 단어를 보지 못하도록 마스킹한다. 번역 예시에서 "나는"을 예측할 때 "NLP를", "좋아한다"는 아직 보면 안 되므로 해당 위치를 $-\infty$로 마스킹해서 Softmax 후 확률이 0이 되게 한다.

### Feed Forward + Add & Norm

각 Attention 층 다음에 위치하는 완전 연결 신경망이다. 두 개의 Dense 층으로 구성되고, Attention이 포착한 관계 정보를 비선형 변환으로 더 풍부하게 처리한다.

Add & Norm은 잔차 연결(Residual Connection)과 Layer Normalization을 합친 것이다. 입력을 그대로 더해줘서 기울기가 직접 흐르는 경로를 확보하고, 정규화로 학습을 안정화한다.

$$\text{출력} = \text{LayerNorm}(x + \text{Sublayer}(x))$$

### GPT-3는 12288

GPT-3의 은닉 차원 크기(d_model)가 12,288이라는 의미다.

```
GPT-3 규모:
  d_model (은닉 차원):  12,288
  레이어 수:            96
  Attention Head 수:    96
  전체 파라미터:        1,750억 개 (175B)

비교:
  BERT-base: d_model=768, 파라미터=110M
  GPT-2:     d_model=1600, 파라미터=1.5B
  GPT-3:     d_model=12288, 파라미터=175B
  GPT-4:     미공개 (추정 수십조 파라미터)
```

---

## 2. Encoder와 Decoder — BERT와 GPT

Transformer의 Encoder와 Decoder를 어떻게 쓰느냐에 따라 모델의 특성이 완전히 달라진다.

### Encoder 전용 — BERT

**BERT (Bidirectional Encoder Representations from Transformers)** 는 Transformer의 Encoder 부분만 사용한다. 문장 전체를 양방향으로 동시에 읽어서 **문맥을 깊이 이해**하는 데 특화되어 있다.

```
BERT의 학습 방식:

1. Masked Language Model (MLM):
   "나는 [MASK]를 먹었다" → "[MASK] = 밥"을 맞추는 학습
   → 앞뒤 문맥을 모두 보고 빈칸 예측

2. Next Sentence Prediction (NSP):
   문장 A와 B가 실제로 이어지는지 판별

→ 양방향 문맥 이해가 뛰어나다
```

BERT의 주요 활용:

- 문서 분류 (감성 분석, 스팸 탐지)
- 개체명 인식 (NER)
- 질의응답 (Q&A)
- 문장 유사도 측정

```python
from transformers import BertTokenizer, BertModel
import torch

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
model     = BertModel.from_pretrained('bert-base-multilingual-cased')

text   = "자연어 처리는 재미있다"
inputs = tokenizer(text, return_tensors='pt')
outputs = model(**inputs)

# 문장 전체 표현 (CLS 토큰)
cls_embedding = outputs.last_hidden_state[:, 0, :]
print(cls_embedding.shape)  # (1, 768)
```

### Decoder 전용 — GPT

**GPT (Generative Pre-trained Transformer)** 는 Transformer의 Decoder 부분만 사용한다. 왼쪽에서 오른쪽으로 단방향으로 읽으면서 **다음 단어를 예측**하는 방식으로 학습한다.

```
GPT의 학습 방식 (Causal Language Model):

입력: "나는 밥을"
예측: "먹었다"

이전 단어들만 보고 다음 단어를 맞추는 학습
→ 텍스트 생성에 최적화됨
```

GPT의 주요 활용:

- 텍스트 생성 (글쓰기, 코드 작성)
- 대화 시스템 (ChatGPT)
- 요약, 번역

### Encoder-Decoder 결합 — T5, BART

입력을 Encoder로 이해하고 Decoder로 새로운 텍스트를 생성하는 구조다.

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer

model     = T5ForConditionalGeneration.from_pretrained('t5-small')
tokenizer = T5Tokenizer.from_pretrained('t5-small')

inputs  = tokenizer("summarize: " + long_text, return_tensors='pt',
                    max_length=512, truncation=True)
outputs = model.generate(inputs.input_ids, max_length=150)
summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
```

### Encoder vs Decoder 비교

|구분|Encoder (BERT 계열)|Decoder (GPT 계열)|Enc-Dec (T5/BART)|
|---|---|---|---|
|Attention 방향|양방향|단방향 (왼→오)|혼합|
|특기|문맥 이해|텍스트 생성|번역, 요약|
|학습 방식|MLM|CLM (Next Token)|Seq2Seq|
|대표 모델|BERT, RoBERTa|GPT-2/3/4, Claude|T5, BART|
|주 사용처|분류, NER, QA|생성, 대화|번역, 요약|

---

## 3. Agent와 Agentic — 자율적 문제 해결

### Agent (에이전트)

Agent는 **역할, 목표, 도구(Tools)를 가진 독립적인 작업자** 소프트웨어 시스템이다. 단순히 질문에 답하는 것을 넘어서, 목표를 달성하기 위해 **스스로 계획을 세우고 도구를 선택해서 실행**하는 자율적 시스템이다.

```
일반 LLM 사용:
  사용자: "파이썬으로 웹 스크래핑 코드 작성해줘"
  LLM:   코드 반환 (끝)

Agent 사용:
  사용자: "오늘 날씨 정보를 가져와서 정리해줘"
  Agent:
    1. 계획: 날씨 API 호출 → 데이터 파싱 → 정리
    2. 도구 선택: weather_api_tool
    3. 실행: API 호출
    4. 검토: 응답 확인
    5. 수정: 형식 변환
    6. 반환: 정리된 날씨 정보
```

### Agentic (에이전틱)

여러 단계에 걸쳐 **스스로 계획 → 실행 → 검토 → 수정**을 반복하는 "자율적 해결 프로세스"를 의미한다. 단일 Agent가 아니라 이런 동작 방식 자체를 가리키는 개념이다.

```
Agentic 프로세스:

목표 설정
    ↓
계획 수립 (어떤 순서로 무엇을 할지)
    ↓
도구 실행 (API 호출, 코드 실행, 검색 등)
    ↓
결과 검토 (의도한 결과인가?)
    ↓ No → 수정 계획 → 다시 실행
    ↓ Yes
최종 결과 반환

→ 이 루프를 목표 달성까지 자율적으로 반복
```

### Agent의 주요 도구 (Tools)

```
검색 도구:      인터넷 검색, 문서 검색
코드 실행:      Python 코드 실행, 결과 확인
API 호출:       날씨, 주식, 지도, 데이터베이스
파일 처리:      파일 읽기/쓰기, PDF 파싱
계산기:         수학 계산
브라우저 조작:  웹 페이지 탐색, 폼 제출
```

---

## 4. LangChain — 범용 LLM 플랫폼

LangChain은 LLM을 이용한 애플리케이션을 쉽게 만들 수 있는 **범용 프레임워크**다. LLM 하나만으로 해결하기 어려운 복잡한 작업을 **Agent + Tools + Memory + RAG** 조합으로 구성할 수 있다.

```
LangChain 구성 요소:

LLM / ChatModel     → GPT-4, Claude, Gemini 등 모델 연결
Prompt Template     → 입력 형식 표준화
Memory              → 대화 이력 저장 및 참조
Tools               → 검색, 코드 실행, API 등 외부 기능
Agent               → 도구를 선택하고 실행하는 자율 주체
RAG                 → 외부 문서에서 정보를 검색해서 답변에 활용
Chain               → 여러 단계를 파이프라인으로 연결
```

### 핵심 구성 요소 상세

**Chain**: 여러 처리 단계를 순서대로 연결하는 파이프라인이다.

```python
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser

llm    = ChatOpenAI(model='gpt-4o', temperature=0)
prompt = ChatPromptTemplate.from_template("다음을 한국어로 번역해줘: {text}")
chain  = prompt | llm | StrOutputParser()

result = chain.invoke({"text": "Attention is all you need"})
print(result)   # "어텐션이 전부다"
```

**Memory**: 대화 이력을 저장해서 이전 맥락을 참고할 수 있게 한다.

```python
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain

memory = ConversationBufferMemory()
conversation = ConversationChain(llm=llm, memory=memory)

conversation.predict(input="내 이름은 수용이야")
response = conversation.predict(input="내 이름이 뭐야?")
print(response)   # "수용이라고 하셨습니다."
```

**RAG (Retrieval Augmented Generation)**: 외부 문서를 검색해서 LLM이 학습하지 않은 최신 정보나 내부 문서에 대해 답변할 수 있게 한다.

```
RAG 흐름:

사용자 질문
    ↓
질문을 벡터로 변환 (Embedding)
    ↓
벡터 DB에서 유사한 문서 검색
    ↓
검색된 문서 + 원래 질문을 합쳐서 LLM에 전달
    ↓
LLM이 문서를 참고해서 답변 생성
```

```python
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# 문서 분할
texts = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200
).split_text(document_text)

# 벡터 저장소 생성
embeddings  = OpenAIEmbeddings()
vectorstore = FAISS.from_texts(texts, embeddings)

# RAG 체인 구성
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
    return_source_documents=True
)

result = qa_chain.invoke({"query": "이 문서의 핵심 내용은?"})
print(result['result'])
```

---

## 5. LangGraph — 상태머신형 Agent 흐름 관리 도구

LangChain이 선형 파이프라인(A → B → C)에 강하다면, LangGraph는 **조건 분기, 루프, 병렬 처리**가 있는 복잡한 Agent 흐름을 관리하는 도구다. 상태머신(State Machine) 개념을 기반으로 한다.

### 핵심 개념

**Graph (그래프)**: 노드(Node)와 엣지(Edge)로 Agent 흐름을 표현한다. 노드는 실행 단위(LLM 호출, 도구 실행, 사람 검토 등)이고, 엣지는 노드 간의 연결(다음에 무엇을 실행할지)이다.

**State (상태)**: 그래프 실행 전체에 걸쳐 공유되는 데이터 저장소다. 각 노드는 State를 읽고 업데이트하면서 정보를 주고받는다.

**Cycles (순환 루프)**: LangChain의 일반 Chain은 한 방향으로만 흐르지만, LangGraph는 조건에 따라 이전 노드로 돌아가는 루프를 만들 수 있다. "결과가 충분하지 않으면 다시 검색"같은 반복 로직을 구현할 수 있다.

```
LangGraph 흐름 예시 (ReAct Agent):

START
  ↓
[LLM 노드] 계획 수립
  ↓
[조건 엣지] 도구 필요?
  ├─ Yes → [도구 실행 노드] → 결과 확인 → [LLM 노드] (루프)
  └─ No  → [최종 답변 노드]
              ↓
            END
```

```python
from langgraph.graph import StateGraph, END
from typing import TypedDict, Annotated
import operator

# State 정의
class AgentState(TypedDict):
    messages:   Annotated[list, operator.add]  # 메시지 이력 누적
    tool_calls: list                            # 도구 호출 기록
    final_answer: str                           # 최종 답변

# 노드 정의
def llm_node(state: AgentState) -> AgentState:
    """LLM이 다음 행동을 결정"""
    response = llm.invoke(state['messages'])
    return {"messages": [response]}

def tool_node(state: AgentState) -> AgentState:
    """도구를 실행하고 결과를 State에 저장"""
    results = []
    for tool_call in state['messages'][-1].tool_calls:
        tool   = tools_by_name[tool_call['name']]
        result = tool.invoke(tool_call['args'])
        results.append(result)
    return {"tool_calls": results}

def should_continue(state: AgentState) -> str:
    """다음 노드를 결정하는 조건 함수"""
    last_message = state['messages'][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tool"   # 도구 실행으로 이동
    return "end"        # 종료

# 그래프 구성
graph = StateGraph(AgentState)
graph.add_node("llm",  llm_node)
graph.add_node("tool", tool_node)

graph.set_entry_point("llm")
graph.add_conditional_edges(
    "llm",
    should_continue,
    {"tool": "tool", "end": END}
)
graph.add_edge("tool", "llm")   # 도구 실행 후 LLM으로 복귀 (루프)

app = graph.compile()
result = app.invoke({"messages": [HumanMessage(content="오늘 서울 날씨는?")]})
```

---

## 6. LlamaIndex — RAG 전용 프레임워크

LlamaIndex(구 GPT Index)는 **데이터 검색과 RAG 성능 극대화**에 특화된 프레임워크다. LangChain이 범용 LLM 플랫폼이라면, LlamaIndex는 "내 데이터를 LLM이 잘 활용하도록 만드는 것"에 집중한다.

### 핵심 기능

**데이터 검색**: PDF, Word, CSV, 웹 페이지, 데이터베이스 등 다양한 소스에서 데이터를 읽어와서 LLM이 질문에 답할 수 있는 형태로 변환한다.

**인덱싱**: 문서를 어떻게 분할하고 저장할지를 결정한다. 검색 성능이 인덱싱 방법에 크게 영향을 받는다.

**다양한 인덱스 종류**:

```
Vector Index (기본):
  문서를 벡터로 변환 후 코사인 유사도로 검색
  → 의미적으로 유사한 내용 검색에 강함

Tree Index:
  문서를 계층 트리로 구성 후 루트에서 리프까지 탐색
  → 긴 문서의 요약 및 계층 탐색에 유리

Summary Index:
  문서 전체 요약본을 인덱스로 만들어 저장
  → 전체 내용을 먼저 파악해야 할 때 유용

Keyword Index:
  키워드 기반 역인덱스 (전통적인 검색 방식)
  → 정확한 단어 매칭이 필요할 때 사용
```

```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core import Settings
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.openai import OpenAIEmbedding

# 모델 설정
Settings.llm       = OpenAI(model='gpt-4o', temperature=0)
Settings.embed_model = OpenAIEmbedding(model='text-embedding-3-small')

# 문서 로드 및 인덱스 생성
documents = SimpleDirectoryReader('data/').load_data()
index     = VectorStoreIndex.from_documents(documents)

# 검색 엔진 구성
query_engine = index.as_query_engine(
    similarity_top_k=3,          # 유사한 문서 3개 검색
    response_mode='compact'       # 응답 형식
)

# 질문
response = query_engine.query("이 문서에서 Transformer를 어떻게 설명하는가?")
print(response)
print(response.source_nodes)   # 참고한 문서 출처
```

### LlamaIndex vs LangChain RAG 비교

|구분|LlamaIndex|LangChain RAG|
|---|---|---|
|목적|RAG 전용, 데이터 중심|범용 LLM 파이프라인|
|인덱싱 종류|Vector, Tree, Summary, Keyword 등 다양|주로 Vector|
|데이터 소스|PDF, DB, API 등 80+ 커넥터|기본 제공 + 커스텀|
|검색 최적화|HyDE, 재순위화, 앙상블 등 고급 기법|기본 검색|
|학습 곡선|중간|낮음|
|적합한 경우|대규모 문서 기반 Q&A|다양한 LLM 태스크|

---

## 7. LLM 생태계 전체 구조

지금까지 살펴본 개념들이 실제로 어떻게 연결되는지 전체 흐름으로 정리한다.

```
[LLM 기반 애플리케이션 스택]

사용자 인터페이스
        ↓
   Agent 레이어 (LangGraph)
   ┌────────────────────────────────────┐
   │  목표 설정 → 계획 → 실행 → 검토 → 수정 │
   └────────────────────────────────────┘
        ↓                    ↓
   LLM 모델             Tools & Actions
   (GPT, Claude,         (검색, 코드 실행,
    Gemini 등)            API 호출 등)
        ↓
   LangChain             LlamaIndex
   (범용 파이프라인)      (RAG 전용)
        ↓                    ↓
   Vector DB (FAISS, Pinecone, Chroma 등)
        ↓
   외부 문서 / 데이터베이스
```

### 도구 선택 가이드

```
무엇을 만들고 싶은가?

단순 LLM 호출 → OpenAI API 직접 사용
텍스트 분류, NER → BERT 계열 (HuggingFace)
텍스트 생성, 대화 → GPT/Claude API
문서 기반 Q&A → LlamaIndex (RAG 전용)
복잡한 LLM 파이프라인 → LangChain
조건 분기/루프가 있는 Agent → LangGraph
여러 Agent 협업 → CrewAI, AutoGen
```

---

## 8. 핵심 요약

```
Transformer 핵심:
  Encoder:  양방향 문맥 이해 → BERT (분류, NER, QA)
  Decoder:  단방향 텍스트 생성 → GPT (생성, 대화)
  Enc-Dec:  입력 이해 + 출력 생성 → T5, BART (번역, 요약)

  Multi-Head Attention: 여러 관점으로 단어 간 관계 동시 포착
  Positional Encoding:  순서 정보를 임베딩에 추가
  GPT-3 d_model = 12,288, 파라미터 175B

Agent & Agentic:
  Agent = 역할 + 목표 + 도구를 가진 자율 소프트웨어
  Agentic = 계획 → 실행 → 검토 → 수정을 자율 반복

LLM 프레임워크:
  LangChain  → 범용 LLM 플랫폼 (Agent + Tools + Memory + RAG)
  LangGraph  → 상태머신 기반 복잡한 Agent 흐름 (Graph + State + Cycles)
  LlamaIndex → RAG 전용 (Vector/Tree/Summary/Keyword 인덱스)
```