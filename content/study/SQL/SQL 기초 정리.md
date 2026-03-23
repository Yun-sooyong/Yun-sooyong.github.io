---
title: SQL 기초 정리 
tags: 
  - SQL
  - database
  - 기초
---

# SQL 기초
> SQL은 데이터베이스에서 데이터를 조회, 정리, 분석할 때 사용하는 언어
---
##  SELECT : 기본 조회
```SQL
-- patient 테이블의 전체 열을 가져오기 
SELECT * 
FROM patients;

-- 지정된 열만 가져오기 
SELECT patient_id, name, age 
FROM patients;
```
- SELECT : 가져올 열 지정
- \* : 전부 지정
- FROM : 데이터를 가져올 테이블

> [!tip] SQL 문은 문장 종료 시 ;(세미콜론)을 붙여야 문장 끝을 인식함
---
## WHERE : 조건 조회
```SQL
-- 단일 조건 
-- patients 테이블에서 age가 60이 넘는 데이터 전체를 가져옴
SELECT *
FROM patients
WHERE age >= 60;

-- 여러 조건
SELECT *
FROM patients
WHERE age >= 60 AND gender = 'M';
```
- AND, OR, IN, NOT, BETWEEN, IS NULL 같은 논리 연산자들이 존재
---

## ORDER BY : 정렬
```SQL 
-- 오름차순
SELECT *
FROM patients
ORDER BY age;

-- 내림차순
SELECT *
FROM patients
ORDER BY age DESC;
```
---

## LIMIT : 개수 제한 
```SQL
-- patients 테이블에서 위에서 10줄 까지만 가져옴 
SELECT *
FROM patients
LIMIT 10;
```
---

##  COUNT / AVG / MAX / MIN : 집계 함수 
```SQL
-- COUNT 
-- patients 테이블의 전체 행의 개수를 반환
SELECT COUNT(*)
FROM patients;

-- AVG
-- age의 평균을 반환
SELECT AVG(age)
FROM patients;

-- MAX / MIN 
-- 최대, 최소값을 반환
SELECT MAX(age), MIN(age)
FROM patients;
```
---

## GROUP BY : 그룹화
```SQL
-- 성별을 기준으로 그룹화 하고 해당 성별에 있는 사람 수을 셈
SELECT gender, COUNT(*) AS patient_count 
FROM patients
GROUP BY gender;

-- 성별 별로 평균 나이를 구함 
SELECT gender, AVG(age) AS avg_age
FROM patients
GROUP BY gender; 
```