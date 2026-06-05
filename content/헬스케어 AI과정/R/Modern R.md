---
title: 020 Modern R
tag: 
  - 헬스케어 ai
  - R
description: 260424 수업 내용 정리
---

# Modern R: 데이터 변환 및 전처리 가이드

tidyverse 생태계를 중심으로 데이터를 불러오고, 가공하고, 시각화하는 현대적인 R 작성 방식을 정리한 문서다.

---

## 1. 파이프 연산자

파이프 연산자는 데이터 처리 과정을 하나의 흐름으로 연결하는 문법이다.  
함수를 중첩해서 쓰는 대신, 왼쪽에서 오른쪽으로 읽히는 자연스러운 흐름으로 코드를 작성할 수 있다.

### `%>%` (기본 파이프)

왼쪽의 결과를 오른쪽 함수의 **첫 번째 인자**로 전달한다.

```r
x %>% f(y)   # f(x, y)와 동일
```

```r
# 중첩 방식 (읽기 어려움)
round(mean(c(1, 2, NA, 3), na.rm = TRUE), 2)

# 파이프 방식 (읽기 쉬움)
c(1, 2, NA, 3) %>%
  mean(na.rm = TRUE) %>%
  round(2)
```

**위치 지정**: 첫 번째 인자가 아닌 다른 위치에 넣고 싶으면 `.`을 사용한다.

```r
x %>% sin(.)
```

**스타일 규칙**: 연산자 앞에는 공백, 뒤에는 줄바꿈을 권장한다. 들여쓰기를 맞추면 흐름이 더 잘 읽힌다.

### `%<>%` (할당 파이프)

연산 결과를 왼쪽 변수에 **즉시 다시 저장**한다. 변수를 가공한 뒤 같은 이름으로 덮어쓸 때 유용하다.

```r
x %<>% abs %>% sort

# 위 코드는 아래와 같다
x <- x %>% abs %>% sort
```

### `%T>%` (티 연산자)

중간 결과를 **출력하거나 시각화**하면서, 데이터 흐름을 끊지 않고 원본 데이터를 그대로 다음 함수로 넘긴다.

```r
x %>%
  abs %>%
  %T>% print %>%   # 중간에 출력, 흐름은 계속
  sort
```

디버깅이나 파이프 중간에 플롯을 삽입할 때 쓴다.

---

## 2. 티블 (Tibble)

기존 `data.frame`을 개선한 구조다. 대용량 데이터를 다룰 때 더 안정적이고 출력도 깔끔하다.

```r
library(tibble)
as_tibble(df)   # data.frame을 tibble로 변환
```

### data.frame과의 차이

|항목|data.frame|tibble|
|---|---|---|
|출력|전체 출력|상위 10행만 출력, 타입 표시|
|문자열 자동변환|Factor로 자동 변환|변환하지 않음|
|행 이름|지원|지양, 열로 변환해서 관리|
|리스트 열|불편|지원|

**문자열 자동변환 없음**: `data.frame`은 문자열 열을 자동으로 Factor로 바꿔버리는 경우가 있어서 예상치 못한 오류가 생기기 쉽다. tibble은 이 동작을 하지 않는다.

**행 이름 처리**: 행 이름이 필요한 경우 열 데이터로 변환해서 쓰는 것이 표준이다.

```r
df %>% rownames_to_column("id")
```

---

## 3. dplyr: 데이터 조작

SQL과 비슷한 방식으로 데이터를 선택하고, 필터링하고, 집계한다. 파이프와 함께 쓰면 전처리 흐름 전체를 하나의 코드 블록으로 표현할 수 있다.

### 주요 함수 정리

|dplyr 함수|SQL 대응|설명|
|---|---|---|
|`select()`|SELECT|필요한 열 선택|
|`filter()`|WHERE|조건에 맞는 행 추출|
|`arrange()`|ORDER BY|데이터 정렬|
|`mutate()`|-|파생 변수 생성|
|`summarise()`|집계 함수|통계량 산출|
|`group_by()`|GROUP BY|그룹화|
|`rename()`|-|열 이름 변경|

### select — 열 선택

```r
df %>% select(name, score)       # 특정 열만 선택
df %>% select(-id)               # id 열 제외
df %>% select(starts_with("s"))  # "s"로 시작하는 열 선택
```

### filter — 행 필터링

```r
df %>% filter(score >= 80)              # 80점 이상
df %>% filter(score >= 80, age < 30)    # AND 조건 (쉼표로 연결)
df %>% filter(grade == "A" | grade == "B")  # OR 조건
```

### arrange — 정렬

```r
df %>% arrange(score)          # 오름차순
df %>% arrange(desc(score))    # 내림차순
df %>% arrange(-score)         # 내림차순 (단축)
```

### mutate — 파생 변수 생성

기존 열을 가공하거나 새 열을 추가한다.

```r
df %>% mutate(
  total = kor + mat + eng,
  avg   = total / 3
)
```

### summarise + group_by — 그룹별 집계

`group_by()`는 그룹을 나누기만 하고, 실제 계산은 `summarise()`와 조합할 때 이루어진다.

```r
df %>%
  group_by(grade) %>%
  summarise(
    count    = n(),
    avg_score = mean(score, na.rm = TRUE)
  )
```

`n()`은 해당 그룹의 행 수를 반환한다.

### rename — 열 이름 변경

`새이름 = 기존이름` 형식으로 쓴다.

```r
df %>% rename(점수 = score, 이름 = name)
```

### 샘플링 및 상위 데이터

```r
sample_n(df, 10)               # 무작위 10행 추출
top_n(df, 5, score)            # score 기준 상위 5행
```

---

## 4. tidyr: 데이터 구조 재구성

**깔끔한 데이터(Tidy Data)** 원칙에 맞게 데이터 형태를 변환한다.

깔끔한 데이터의 원칙:

- 한 행 = 하나의 관측치
- 한 열 = 하나의 변수

### pivot_longer — 넓은 → 긴 형태

여러 열을 하나의 `feature`와 `value` 열로 모아 **세로로 길게** 만든다.

```r
df %>%
  pivot_longer(
    cols     = c(kor, mat, eng),
    names_to = "subject",
    values_to = "score"
  )
```

**시각화 전 필수 단계**다. ggplot2는 x축에 하나의 열, y축에 하나의 열을 받기 때문에, 여러 변수를 한 번에 비교하려면 먼저 길게 변환해야 한다.

```
변환 전:
| name | kor | mat | eng |
|------|-----|-----|-----|
| Kim  | 80  | 90  | 70  |

변환 후:
| name | subject | score |
|------|---------|-------|
| Kim  | kor     | 80    |
| Kim  | mat     | 90    |
| Kim  | eng     | 70    |
```

### pivot_wider — 긴 → 넓은 형태

세로 데이터를 **가로로 넓게** 펼친다. 보고서용 표를 만들 때 유용하다.

```r
df %>%
  pivot_wider(
    names_from  = subject,
    values_from = score
  )
```

### separate — 열 분리

한 열에 담긴 정보를 구분자 기준으로 여러 열로 쪼갠다.

```r
df %>% separate(date, into = c("year", "month", "day"), sep = "-")
```

### unite — 열 합치기

여러 열의 정보를 하나로 합친다.

```r
df %>% unite("full_name", first, last, sep = " ")
```

---

## 5. 실무 전처리 패턴

### 결측치(NA) 처리

**전체 현황 확인**:

```r
df %>% summarise(across(everything(), ~sum(is.na(.))))
```

**숫자형 열만 확인**:

```r
df %>% summarise(across(where(is.numeric), ~sum(is.na(.))))
```

**평균 또는 중위수로 대체**:

```r
df %>%
  mutate(across(where(is.numeric), ~replace_na(., mean(., na.rm = TRUE))))
```

`na.rm = TRUE` 옵션 없이 `mean()`을 쓰면 결측값이 있을 때 결과가 NA로 나온다. 반드시 명시해야 한다.

### 이상치(Outlier) 탐지 — IQR 방식

```r
df %>%
  mutate(
    Q1      = quantile(score, 0.25),
    Q3      = quantile(score, 0.75),
    IQR_val = Q3 - Q1,
    is_outlier = score < (Q1 - 1.5 * IQR_val) | score > (Q3 + 1.5 * IQR_val)
  )
```

이상치 여부를 `TRUE/FALSE` 변수로 따로 만들어두면 필터링이나 시각화에 바로 활용할 수 있다.

### across — 여러 열에 동일한 함수 적용

같은 함수를 여러 열에 반복 적용할 때 `across()`를 쓰면 코드가 크게 줄어든다.

```r
# across 없이
df %>% summarise(
  kor_mean = mean(kor, na.rm = TRUE),
  mat_mean = mean(mat, na.rm = TRUE),
  eng_mean = mean(eng, na.rm = TRUE)
)

# across 사용
df %>%
  summarise(across(c(kor, mat, eng), ~round(mean(.x, na.rm = TRUE), 2)))
```

`.x`는 현재 처리 중인 열을 가리킨다.

---

## 6. 고급 기능

### purrr — 함수형 프로그래밍

`for` 반복문 대신 `map()` 계열 함수를 사용해 리스트나 벡터의 각 요소에 함수를 적용한다.

```r
library(purrr)

map(list_data, mean)       # 각 요소에 mean 적용, 리스트 반환
map_dbl(list_data, mean)   # 각 요소에 mean 적용, 실수 벡터 반환
```

`map()`은 항상 리스트를 반환한다. 결과 타입을 고정하고 싶을 때는 `map_dbl()`, `map_chr()`, `map_lgl()` 등 타입별 변형을 사용한다.

### forcats — 범주형 순서 관리

그래프에서 막대 순서를 바꾸면 전달력이 크게 달라진다. 기본적으로 알파벳 순서로 정렬되는데, `forcats`를 쓰면 의미 있는 순서로 바꿀 수 있다.

```r
library(forcats)

df %>% mutate(grade = fct_infreq(grade))        # 빈도 높은 순서
df %>% mutate(grade = fct_reorder(grade, score)) # score 크기 순서
```

### RMariaDB — 외부 데이터베이스 연동

```r
library(RMariaDB)

con <- dbConnect(MariaDB(), host = "...", user = "...", password = "...", dbname = "...")

df <- dbReadTable(con, "table_name") %>% as_tibble()  # 로드 후 tibble로 변환

dbWriteTable(con, "result_table", result_df)          # 분석 결과 저장

dbDisconnect(con)   # 연결 해제
```

---

## 7. Modern R 핵심 흐름 정리

실무에서 가장 많이 쓰는 파이프 패턴이다.

```r
df %>%
  rownames_to_column("id") %>%       # 행 이름 → 열로 변환
  filter(!is.na(score)) %>%          # 결측값 제거
  mutate(total = kor + mat + eng) %>% # 파생 변수 생성
  group_by(grade) %>%                # 그룹화
  summarise(avg = mean(total))       # 집계
```

시각화 전 흐름:

```r
df %>%
  pivot_longer(cols = c(kor, mat, eng), names_to = "subject", values_to = "score") %>%
  ggplot(aes(x = subject, y = score)) +
  geom_boxplot()
```

데이터 로드부터 시각화까지 파이프로 연결하는 것이 Modern R의 정석이다.