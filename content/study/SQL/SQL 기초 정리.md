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
