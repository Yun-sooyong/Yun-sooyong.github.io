---
title: 015 Flask
tag: 
  - 국비교육
  - flask
description: 260415 수업 내용 정리
---


# Flask 정리  
  
## 1. Flask란?  
  
Flask는 **Python 기반의 경량 웹 애플리케이션 프레임워크**이며,    
여기서는 **Application Server** 역할로 정리할 수 있다.  


주요 역할은 다음과 같다.  
- 클라이언트 요청(Request) 수신  
- URL 라우팅(Mapping / Routing)  
- 비즈니스 로직 처리  
- 템플릿 렌더링  
- JSON 응답 처리  
- 세션 / 쿠키 관리  
- DB 연동  
  
---  
  
## 2. 전체 구조  
  
> Frontend <-> Backend(Flask) <-> Business Logic <-> DAO <-> DB

### 구성 요소

#### Frontend

- 사용자 화면(UI)
- HTML, CSS, JavaScript 기반
- 브라우저의 DOM(Document Object Model) 구조로 동작

#### Backend

- Flask 서버가 요청과 응답을 처리
- 클라이언트의 `GET`, `POST` 요청을 받음
- 필요에 따라 `PUT`, `DELETE`까지 포함한 RESTful 방식으로 확장 가능

#### Business Logic

- 실제 서비스 로직 처리
- 예: 회원 조회, 게시글 등록, 로그인 검증 등

#### DAO

- **Data Access Object**
- DB에 접근하여 데이터를 조회/저장/수정/삭제하는 계층

#### DB

- 애플리케이션의 실제 데이터 저장소

---

## 3. 요청/응답 흐름

## 3-1. 기본 흐름

1. 사용자가 Frontend에서 요청 발생
2. 브라우저가 Flask 서버로 HTTP 요청 전송
3. Flask가 URL에 맞는 라우팅 처리
4. 필요한 비즈니스 로직 수행
5. DAO를 통해 DB 접근
6. 처리 결과를 Flask가 응답(Response)으로 반환
7. Frontend가 결과를 화면에 표시

---

## 4. HTTP 방식

### 기본 요청 방식

- `GET`: 데이터 조회
- `POST`: 데이터 전송 / 생성

### RESTful 확장 방식

- `GET`: 조회
- `POST`: 생성
- `PUT`: 수정
- `DELETE`: 삭제

---

## 5. Flask의 주요 기능

## 5-1. 라우팅(Routing)

Flask는 URL 요청을 적절한 함수에 연결한다.

예:

- `/` -> 메인 페이지
- `/login` -> 로그인 처리
- `/users` -> 사용자 목록 조회

---

## 5-2. 템플릿 처리

Flask는 **Jinja 템플릿 엔진**을 사용하여 HTML을 동적으로 생성할 수 있다.

대표 함수:

render_template("index.html", data=data)

설명:

- `index.html` 템플릿을 렌더링
- `data` 값을 HTML로 전달

---

## 5-3. 응답(Response) 처리

### HTML 응답

return render_template("index.html", data=data)

### JSON 응답

from flask import jsonify  
  
return jsonify({"result": "success"})

### 페이지 이동

from flask import redirect  
  
return redirect("/home")

---

## 5-4. 쿠키 / 세션 관리

### Cookie

- 클라이언트 브라우저에 저장되는 데이터
- 로그인 유지, 사용자 설정 저장 등에 사용

### Session

- 서버 측에서 사용자 상태를 관리
- 인증 정보, 로그인 상태 유지 등에 사용

보통 Flask에서는 다음 개념으로 이해하면 된다.

- **Cookie**: 클라이언트에 저장
- **Session**: 서버 중심 상태 관리

---

## 6. 데이터 전달 방법

원본 내용 기준 주요 전달 방식은 다음과 같다.

### 1) 템플릿으로 전달

render_template("index.html", data=data)

### 2) 쿠키를 포함한 응답 생성

원본에는 `make_template(Cookie) - response` 형태로 적혀 있으나,  
일반적으로는 아래처럼 **응답 객체를 만든 뒤 쿠키를 설정**하는 형태로 사용한다.
```python
from flask import make_response  
  
response = make_response("OK")  
response.set_cookie("username", "user1")  
return response
```

### 3) JSON 반환

```python
return jsonify(data)
```

### 4) 리다이렉트

```python
return redirect("/next-page")
```

---

## 7. Frontend 관점 정리

예시 구조:
- window
    - history
    - document
    - location

그리고 `document` 아래에 다음과 같은 요소들이 올 수 있다.

- link
- layer
- form
- script
- image

또한 form 내부 입력 요소 예시:

- text
- textarea
- password
- checkbox
- radio

즉, Frontend는 **브라우저 DOM 구조 위에서 사용자 입력을 만들고**,  
그 입력값을 Flask Backend로 전달하는 역할을 한다.

---

## 8. 확장 요소: 크롤링 애플리케이션

원본 하단에는 `crawling Application`이 함께 표시되어 있다.

관련 기술 예시:

- BeautifulSoup
- 정규표현식(RE)
- Selenium
- OpenAPI

의미:
- Flask 애플리케이션은 일반적인 웹 서비스뿐 아니라
- 외부 데이터 수집(크롤링) 또는 Open API 연동 기능과도 연결될 수 있다.
- 수집한 데이터를 DB에 저장하고, 다시 Flask를 통해 사용자에게 제공할 수 있다.

---

## 9. 핵심 요약

### 한 줄 요약

Flask는 **클라이언트 요청을 받아 라우팅하고, 비즈니스 로직과 DB 처리를 거쳐, HTML/JSON/리다이렉트 형태로 응답하는 웹 애플리케이션 서버**이다.

### 핵심 키워드

- Flask
- Routing
- Template(Jinja)
- Request / Response
- Session / Cookies
- Business Logic
- DAO
- DB
- JSON
- Redirect
- Crawling / OpenAPI 연계 가능

---

## 10. 보완해서 이해하면 좋은 점

원본 메모는 개념 키워드 중심이므로, 실제 학습 시 아래 내용을 함께 보면 더 좋다.

### 추가로 필요한 학습 항목

- Flask 앱 생성 방식
- `@app.route()` 사용법
- `request` 객체 활용
- HTML Form 데이터 받기
- DB 연결 방식(SQLAlchemy 등)
- 세션 보안 설정
- REST API 설계 방식