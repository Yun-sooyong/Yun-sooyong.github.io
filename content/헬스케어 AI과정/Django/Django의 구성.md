---
title: 045 django의 구성
tag:
  - 헬스케어 ai
  - django
description: 260623 수업 내용 정리
---

# 장고(Django)의 구성

Django는 **Python 기반의 웹 애플리케이션 프레임워크**다. "배터리 포함(Batteries Included)" 철학으로 웹 개발에 필요한 기능들을 대부분 내장하고 있어 빠르게 실용적인 웹 서비스를 만들 수 있다.

머신러닝 프로젝트에서도 Django가 자주 쓰인다. 학습한 모델을 REST API로 배포하거나, 데이터 관리 어드민 페이지를 만들거나, 예측 결과를 시각화해서 보여주는 대시보드를 구축할 때 활용된다.

**핵심 흐름 한 줄 요약**:

> **요청(request)** → URL Resolver → View → (Model + Template) → **응답(response)**

---

## 1. Django의 주요 특징

|특징|설명|
|---|---|
|**Python 기반**|Cross-platform 지원 (Windows/macOS/Linux)|
|**Web Application Framework**|웹 앱 개발을 위한 풀스택 프레임워크|
|**관리자 페이지 자동 관리**|`admin` 패널 기본 제공|
|**ORM 사용**|Python 코드로 DB 조작 (SQL 직접 작성 불필요)|
|**MVT 구조**|MVC 패턴에 대응하는 Django식 구조|
|**Jinja2 Template Engine**|HTML 템플릿 렌더링에 사용|

각 특징을 상세히 살펴보자.

### Python 기반의 Cross-Platform

Python으로 작성되어 Windows, macOS, Linux 어디서든 동일하게 실행된다. Python 생태계의 NumPy, Pandas, scikit-learn 등과 자연스럽게 연동된다.

### Web Application Framework

프레임워크는 **웹 개발에서 반복적으로 필요한 기능들을 미리 구현해 놓은 뼈대**다. 로그인/로그아웃, 세션 관리, URL 라우팅, 폼 처리, 보안(CSRF, SQL 인젝션 방어) 등을 개발자가 처음부터 구현하지 않아도 된다.

```
프레임워크 없이:          Django 사용 시:
  URL 파싱 직접 구현        urls.py에 패턴만 등록
  DB 연결 직접 처리         models.py에 클래스만 정의
  HTML 조립 직접 코딩       templates에 파일만 작성
  세션 직접 구현            세션 기능 내장
  보안 직접 처리            보안 기능 내장

→ 비즈니스 로직에만 집중 가능
```

### 관리자 페이지 자동 관리 (Admin)

모델(DB 테이블)을 등록하면 **데이터를 웹에서 CRUD(생성/조회/수정/삭제)할 수 있는 관리자 페이지**가 자동으로 생성된다. 슬라이드 상단의 `admin` 박스가 이것이다.

```python
# admin.py
from django.contrib import admin
from .models import Student

admin.site.register(Student)  # 이 한 줄만으로 관리자 페이지에 Student 테이블이 등장
```

`/admin` URL로 접속하면 로그인 후 데이터를 관리할 수 있다. ML 프로젝트에서 훈련 데이터를 관리하거나 예측 결과를 확인할 때 유용하다.

### ORM (Object-Relational Mapping) 사용

SQL을 직접 쓰지 않고 **Python 객체로 데이터베이스를 조작**한다. 슬라이드 오른쪽의 `Model(ORM)` 박스가 이것이다.

```python
# SQL 방식                      Django ORM 방식
"SELECT * FROM student          Student.objects.all()
 WHERE kor > 80"                Student.objects.filter(kor__gt=80)

"INSERT INTO student ..."       s = Student(name='홍길동', kor=90)
                                s.save()
```

### MVT 구조 (MVC의 Django 버전)

일반적인 MVC(Model-View-Controller) 패턴을 Django에서는 **MVT(Model-View-Template)** 으로 부른다.

|MVC|MVT (Django)|역할|
|---|---|---|
|Model|**Model**|데이터와 비즈니스 로직|
|View|**Template**|화면 출력 (HTML)|
|Controller|**View**|요청 처리, 흐름 제어|

### Jinja Template Engine

HTML에 Python 문법을 삽입할 수 있는 템플릿 엔진이다. Django는 기본적으로 DTL(Django Template Language)을 사용하고 Jinja2와도 호환된다.

```html
<!-- templates/students.html -->
<ul>
  {% for student in students %}
    <li>{{ student.name }}: 국어 {{ student.kor }}점</li>
  {% endfor %}
</ul>
```

---

## 2. Django의 MVT 구조와 요청 처리 흐름

슬라이드의 중앙 다이어그램이 보여주는 전체 흐름이다.

```
브라우저                      Django 서버
    │                              │
    │── request ──────────────────→│
    │              URL Resolver     │  urls.py에서 패턴 매칭
    │              ↓               │
    │              View             │  views.py에서 비즈니스 로직 처리
    │              ↓↑              │
    │              Model (ORM)      │  models.py → DB 조회/저장
    │              ↓               │
    │              Template         │  templates/xxx.html 렌더링
    │← response ───────────────────│
    │  (HTML/JSON/Redirect 등)     │
```

각 구성 요소의 역할을 한눈에 정리하면 다음과 같다.

|구성 요소|역할|
|---|---|
|**URL Resolver**|URL 패턴 매핑, 적절한 View 호출|
|**View**|비즈니스 로직 처리 (CBV / FBV)|
|**Model (ORM)**|데이터베이스 테이블을 Python 클래스로 표현|
|**Template**|사용자에게 보여줄 HTML 생성 (Jinja2 / DTL)|
|**Admin**|모델 기반 자동 관리자 페이지|

### URL Resolver — 요청을 올바른 View로 연결

브라우저가 URL을 요청하면 `urls.py`의 패턴과 비교해서 어떤 View 함수를 실행할지 결정한다.

```python
# urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('',              views.index,         name='index'),
    path('students/',     views.student_list,  name='student-list'),
    path('students/<int:pk>/', views.student_detail, name='student-detail'),
    path('admin/',        admin.site.urls),
]
```

```
요청: GET /students/3/
  → urlpatterns를 위에서부터 순서대로 비교
  → 'students/<int:pk>/'에 매칭, pk=3
  → views.student_detail(request, pk=3) 실행
```

### View — 비즈니스 로직 처리

요청을 받아 Model에서 데이터를 가져오고 Template에 넘기는 중간 처리자다.

```python
# views.py
from django.shortcuts import render
from .models import Student

def student_list(request):
    # 1. Model에서 데이터 조회
    students = Student.objects.all().order_by('-kor')  # 국어 점수 내림차순
    
    # 2. Template에 데이터 전달
    context = {'students': students}
    return render(request, 'students/list.html', context)
```

### Response 종류

슬라이드에 표시된 응답 유형들:

```python
from django.http import HttpResponse, JsonResponse
from django.shortcuts import redirect

# HttpResponse: 직접 문자열 반환
return HttpResponse('<h1>Hello World</h1>')
return HttpResponse(status=404)

# JsonResponse: JSON 데이터 반환 (API 서버)
return JsonResponse({'students': list(qs.values()), 'count': qs.count()})

# redirect: 다른 URL로 이동
return redirect('student-list')
return redirect('students/1/')

# render: Template + Context → HTML 반환 (가장 많이 사용)
return render(request, 'template.html', context)
```

---

## 3. Model (ORM) — 데이터베이스와의 연결

### 슬라이드의 Student 모델

```python
# models.py
class Student(models.Model):
    def __init__(self):
        self.bunho = 0     # 번호
        self.name  = ''    # 이름
        self.kor   = 0     # 국어 점수
        self.mat   = 0     # 수학 점수
        self.eng   = 0     # 영어 점수
```

Django에서는 이보다 더 간결하게 `models.Model`을 상속해서 정의한다.

```python
# models.py (Django 방식)
from django.db import models

class Student(models.Model):
    bunho = models.AutoField(primary_key=True)  # 자동 증가 PK (INT)
    name  = models.CharField(max_length=50)      # 이름 (VARCHAR)
    kor   = models.SmallIntegerField(default=0)  # 국어 (TINYINT)
    mat   = models.SmallIntegerField(default=0)  # 수학 (TINYINT)
    eng   = models.SmallIntegerField(default=0)  # 영어 (TINYINT)

    class Meta:
        db_table = 'student'   # DB에서 사용할 테이블 이름
        ordering = ['bunho']   # 기본 정렬 기준

    def __str__(self):
        return f'{self.name} (국어:{self.kor}, 수학:{self.mat}, 영어:{self.eng})'
```

슬라이드 하단의 DB 테이블이 Python 클래스로 변환된 것이다.

|컬럼명|타입|대응 필드|
|---|---|---|
|bunho|INT|`AutoField`|
|name|VARCHAR|`CharField`|
|kor|TINYINT|`SmallIntegerField`|
|mat|TINYINT|`SmallIntegerField`|
|eng|TINYINT|`SmallIntegerField`|

### ORM 작동 원리

```
Python 코드           →   SQL 변환                →   MySQL/PostgreSQL 등
Student.objects.all()  →  SELECT * FROM student;   →   DB 조회 결과
Student(name='철수').save() → INSERT INTO student ...  →   DB 저장
```

ORM이 중간에서 번역해주기 때문에 코드를 바꾸지 않고도 MySQL, PostgreSQL, SQLite 등 다른 DB로 전환할 수 있다.

### Migrations — 코드 변경 → DB 변경 자동화

Model 클래스를 수정하면 DB 테이블 구조도 바꿔야 한다. `migrations`가 이 과정을 자동으로 처리한다.

```bash
# 1. 모델 변경사항을 마이그레이션 파일로 생성
python manage.py makemigrations

# 2. 마이그레이션 파일을 실제 DB에 적용
python manage.py migrate
```

```
models.py 수정 (새 필드 추가)
   ↓ makemigrations
0003_student_add_score.py (변경 내역 기록)
   ↓ migrate
DB에 ALTER TABLE student ADD COLUMN score ... 실행
```

### ORM 주요 쿼리

```python
from .models import Student

# ── 조회 ──────────────────────────────────────────────
Student.objects.all()                   # 전체 조회
Student.objects.get(pk=1)               # 단일 조회 (없으면 예외)
Student.objects.filter(kor__gte=80)     # 조건 필터
Student.objects.exclude(name='홍길동')  # 제외
Student.objects.order_by('-kor', 'mat') # 정렬 (- 붙이면 내림차순)

# ── 필터 조건 (Field Lookups) ─────────────────────────
Student.objects.filter(kor__exact=90)   # kor = 90
Student.objects.filter(kor__gt=80)      # kor > 80 (greater than)
Student.objects.filter(kor__gte=80)     # kor >= 80
Student.objects.filter(kor__lt=60)      # kor < 60
Student.objects.filter(name__contains='길')   # LIKE '%길%'
Student.objects.filter(name__startswith='홍') # LIKE '홍%'

# ── 집계 ──────────────────────────────────────────────
from django.db.models import Avg, Max, Min, Count, Sum

Student.objects.aggregate(Avg('kor'))         # {'kor__avg': 75.2}
Student.objects.aggregate(Max('kor'))         # {'kor__max': 98}
Student.objects.values('name').annotate(Count('pk'))  # 이름별 수

# ── 생성/수정/삭제 ─────────────────────────────────────
s = Student.objects.create(name='이순신', kor=95, mat=90, eng=88)
Student.objects.filter(pk=1).update(kor=100)
Student.objects.filter(pk=1).delete()

# ── QuerySet → DataFrame 변환 (ML 연동) ───────────────
import pandas as pd
qs = Student.objects.all().values()    # QuerySet을 딕셔너리 형태로
df = pd.DataFrame(list(qs))           # pandas DataFrame으로 변환
```

### MySQL 연결 설정

슬라이드의 `MySQL` 박스가 ORM과 연결된 부분이다.

```python
# settings.py
DATABASES = {
    'default': {
        'ENGINE':   'django.db.backends.mysql',
        'NAME':     'mydb',       # DB 이름
        'USER':     'root',       # DB 사용자
        'PASSWORD': 'password',   # 비밀번호
        'HOST':     'localhost',  # DB 서버 주소
        'PORT':     '3306',       # MySQL 기본 포트
    }
}
```

```bash
pip install mysqlclient   # MySQL 드라이버 설치
```

---

## 4. Template (Jinja/DTL) — 화면 출력

Template은 View에서 받은 데이터를 HTML로 렌더링하는 역할이다. Django Template Language(DTL)와 Jinja2 두 가지를 지원한다.

### DTL 기본 문법

```html
<!-- templates/students/list.html -->
<!DOCTYPE html>
<html>
<head>
    <title>학생 목록</title>
    <!-- Bootstrap CSS CDN (슬라이드의 Bootstrap) -->
    <link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
</head>
<body>
<div class="container mt-4">
    <h1>학생 목록</h1>

    <!-- 변수 출력: {{ 변수명 }} -->
    <p>총 학생 수: {{ students.count }}명</p>

    <!-- 반복문: {% for %} -->
    <table class="table table-striped">
        <thead>
            <tr><th>번호</th><th>이름</th><th>국어</th><th>수학</th><th>영어</th><th>평균</th></tr>
        </thead>
        <tbody>
        {% for s in students %}
            <!-- 조건문: {% if %} -->
            <tr class="{% if s.kor >= 90 %}table-success{% elif s.kor < 60 %}table-danger{% endif %}">
                <td>{{ forloop.counter }}</td>  <!-- 반복 번호 -->
                <td>{{ s.name }}</td>
                <td>{{ s.kor }}</td>
                <td>{{ s.mat }}</td>
                <td>{{ s.eng }}</td>
                <td>{{ s.kor|add:s.mat|add:s.eng|divisibleby:3 }}</td>  <!-- 필터 -->
            </tr>
        {% empty %}
            <tr><td colspan="6">등록된 학생이 없습니다.</td></tr>
        {% endfor %}
        </tbody>
    </table>
</div>

<!-- Bootstrap JS (슬라이드의 Javascript) -->
<script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>
```

### 템플릿 상속

중복 HTML을 줄이기 위해 공통 레이아웃을 상속한다.

```html
<!-- templates/base.html (기본 레이아웃) -->
<!DOCTYPE html>
<html>
<head>
    <title>{% block title %}Django 앱{% endblock %}</title>
    {% block css %}{% endblock %}
</head>
<body>
    <nav>공통 네비게이션</nav>
    {% block content %}{% endblock %}  <!-- 자식이 채우는 부분 -->
    <footer>공통 푸터</footer>
</body>
</html>

<!-- templates/students/list.html (자식 템플릿) -->
{% extends 'base.html' %}

{% block title %}학생 목록{% endblock %}

{% block content %}
<h1>학생 목록</h1>
<!-- 내용 -->
{% endblock %}
```

### Form — 입력 데이터 처리

슬라이드의 `Template(jinja) form` 박스가 이것이다. HTML 폼 검증과 CSRF 보호를 자동으로 처리한다.

```python
# forms.py
from django import forms
from .models import Student

class StudentForm(forms.ModelForm):
    class Meta:
        model = Student
        fields = ['name', 'kor', 'mat', 'eng']
        widgets = {
            'name': forms.TextInput(attrs={'class': 'form-control', 'placeholder': '이름'}),
            'kor':  forms.NumberInput(attrs={'class': 'form-control', 'min': 0, 'max': 100}),
        }
        labels = {
            'name': '이름', 'kor': '국어', 'mat': '수학', 'eng': '영어'
        }
```

```python
# views.py
def student_create(request):
    if request.method == 'POST':
        form = StudentForm(request.POST)   # POST 데이터로 폼 생성
        if form.is_valid():                # 유효성 검사
            form.save()                    # DB 저장
            return redirect('student-list')
    else:
        form = StudentForm()               # 빈 폼
    return render(request, 'students/form.html', {'form': form})
```

```html
<!-- templates/students/form.html -->
<form method="post">
    {% csrf_token %}  <!-- CSRF 보호 토큰 (필수) -->
    {{ form.as_p }}   <!-- 폼 자동 렌더링 -->
    <button type="submit" class="btn btn-primary">저장</button>
</form>
```

---

## 5. CBV vs FBV

슬라이드 하단의 `CBV vs FBVs`와 `request | Model | form` 부분이다.

### FBV (Function-Based Views) — 함수 기반 뷰

```python
# views.py (FBV 방식)
from django.shortcuts import render, get_object_or_404, redirect
from .models import Student
from .forms import StudentForm

# 목록 조회
def student_list(request):
    students = Student.objects.all()
    return render(request, 'students/list.html', {'students': students})

# 상세 조회
def student_detail(request, pk):
    student = get_object_or_404(Student, pk=pk)
    return render(request, 'students/detail.html', {'student': student})

# 생성
def student_create(request):
    if request.method == 'POST':
        form = StudentForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('student-list')
    else:
        form = StudentForm()
    return render(request, 'students/form.html', {'form': form})

# 수정
def student_update(request, pk):
    student = get_object_or_404(Student, pk=pk)
    if request.method == 'POST':
        form = StudentForm(request.POST, instance=student)  # 기존 데이터로 초기화
        if form.is_valid():
            form.save()
            return redirect('student-detail', pk=pk)
    else:
        form = StudentForm(instance=student)
    return render(request, 'students/form.html', {'form': form})

# 삭제
def student_delete(request, pk):
    student = get_object_or_404(Student, pk=pk)
    if request.method == 'POST':
        student.delete()
        return redirect('student-list')
    return render(request, 'students/confirm_delete.html', {'student': student})
```

### CBV (Class-Based Views) — 클래스 기반 뷰

반복적인 CRUD 패턴을 상속과 믹스인으로 **재사용**한다. 코드 양이 줄고 공통 동작을 쉽게 확장할 수 있다.

```python
# views.py (CBV 방식)
from django.views.generic import (
    ListView, DetailView,
    CreateView, UpdateView, DeleteView
)
from django.urls import reverse_lazy
from .models import Student
from .forms import StudentForm

class StudentListView(ListView):
    model = Student
    template_name = 'students/list.html'
    context_object_name = 'students'
    ordering = ['-kor']
    paginate_by = 10   # 페이지당 10개

class StudentDetailView(DetailView):
    model = Student
    template_name = 'students/detail.html'

class StudentCreateView(CreateView):
    model = Student
    form_class = StudentForm
    template_name = 'students/form.html'
    success_url = reverse_lazy('student-list')

class StudentUpdateView(UpdateView):
    model = Student
    form_class = StudentForm
    template_name = 'students/form.html'
    success_url = reverse_lazy('student-list')

class StudentDeleteView(DeleteView):
    model = Student
    template_name = 'students/confirm_delete.html'
    success_url = reverse_lazy('student-list')
```

```python
# urls.py (CBV와 연결)
urlpatterns = [
    path('students/',       StudentListView.as_view(),   name='student-list'),
    path('students/<int:pk>/', StudentDetailView.as_view(), name='student-detail'),
    path('students/new/',   StudentCreateView.as_view(), name='student-create'),
    path('students/<int:pk>/edit/',   StudentUpdateView.as_view(), name='student-update'),
    path('students/<int:pk>/delete/', StudentDeleteView.as_view(), name='student-delete'),
]
```

### FBV vs CBV 비교

|구분|FBV|CBV|
|---|---|---|
|형태|함수로 View 정의|클래스로 View 정의|
|코드 가독성|흐름이 명확, 이해 쉬움|초반에 상속 구조 파악 필요|
|코드 재사용|낮음 (반복 많음)|높음 (Generic View 상속)|
|유연성|높음 (자유롭게 구현)|약간 낮음 (상속 구조 제약)|
|진입장벽|낮음, 초보자 권장|익숙해지면 사용|
|CRUD 구현|각각 함수 작성|Generic View 상속으로 단순|

CBV에서 공통적으로 다루는 핵심 요소 세 가지: `request`(클라이언트 요청 정보), `Model`(DB 데이터), `form`(사용자 입력 처리). 슬라이드의 `request | Model | form` 박스가 이것이다.

---

## 6. Django REST Framework (DRF) — ML 모델 배포

ML 프로젝트에서 학습된 모델을 API로 제공할 때 DRF를 사용한다.

```bash
pip install djangorestframework
```

```python
# settings.py
INSTALLED_APPS = [
    ...
    'rest_framework',
]

# serializers.py (Model → JSON 변환)
from rest_framework import serializers
from .models import Student

class StudentSerializer(serializers.ModelSerializer):
    class Meta:
        model = Student
        fields = '__all__'

# views.py (API 뷰)
from rest_framework.decorators import api_view
from rest_framework.response import Response
import joblib, numpy as np

# 학습된 ML 모델 로드
ml_model = joblib.load('trained_model.pkl')
scaler   = joblib.load('scaler.pkl')

@api_view(['POST'])
def predict(request):
    """학생 성적 데이터를 받아 등급 예측"""
    data = request.data
    features = np.array([[data['kor'], data['mat'], data['eng']]])
    features_scaled = scaler.transform(features)
    prediction = ml_model.predict(features_scaled)[0]
    probability = ml_model.predict_proba(features_scaled)[0].max()

    return Response({
        'prediction': int(prediction),
        'probability': float(probability),
        'message': f'예측 등급: {prediction}'
    })
```

---

## 7. 프로젝트 구조와 설정

### 디렉토리 구조

```
myproject/
├── manage.py          ← Django 관리 명령어 실행
├── myproject/         ← 프로젝트 설정 디렉토리
│   ├── settings.py    ← 전체 설정 (DB, APPS, 미들웨어 등)
│   ├── urls.py        ← 루트 URL 설정
│   └── wsgi.py        ← 배포 시 웹서버 연결
└── students/          ← 앱 디렉토리
    ├── models.py      ← DB 모델 정의
    ├── views.py       ← 비즈니스 로직
    ├── urls.py        ← 앱 URL 설정
    ├── forms.py       ← 폼 정의
    ├── admin.py       ← 관리자 등록
    ├── migrations/    ← DB 변경 이력
    └── templates/     ← HTML 파일
        └── students/
            ├── list.html
            ├── detail.html
            └── form.html
```

### 프로젝트 시작 명령어

```bash
# 1. Django 설치
pip install django

# 2. 프로젝트 생성
django-admin startproject myproject

# 3. 앱 생성
cd myproject
python manage.py startapp students

# 4. 앱을 settings.py에 등록
# INSTALLED_APPS에 'students' 추가

# 5. 모델 작성 후 마이그레이션
python manage.py makemigrations
python manage.py migrate

# 6. 관리자 계정 생성
python manage.py createsuperuser

# 7. 개발 서버 실행
python manage.py runserver   # http://127.0.0.1:8000/
```

### settings.py 주요 설정

```python
# settings.py
from pathlib import Path
import os

BASE_DIR = Path(__file__).resolve().parent.parent

SECRET_KEY = 'django-insecure-...'   # 보안 키 (운영 시 환경변수로 관리)
DEBUG = True                           # 운영 시 False로 변경

ALLOWED_HOSTS = ['*']                  # 허용 호스트 (운영 시 도메인 지정)

INSTALLED_APPS = [
    'django.contrib.admin',            # 관리자 페이지
    'django.contrib.auth',             # 인증 시스템
    'django.contrib.contenttypes',
    'django.contrib.sessions',         # 세션
    'django.contrib.messages',
    'django.contrib.staticfiles',      # 정적 파일
    'students',                        # 내가 만든 앱
    'rest_framework',                  # DRF (API 사용 시)
]

TEMPLATES = [{
    'BACKEND': 'django.template.backends.django.DjangoTemplates',
    'DIRS': [BASE_DIR / 'templates'],  # 템플릿 경로
    'APP_DIRS': True,
    'OPTIONS': {
        'context_processors': [...],
        'libraries': {},               # Jinja2 사용 시 'jinja2' 추가
    },
}]

STATIC_URL  = '/static/'
MEDIA_URL   = '/media/'
MEDIA_ROOT  = BASE_DIR / 'media'      # 업로드 파일 저장 경로

LANGUAGE_CODE = 'ko-kr'
TIME_ZONE     = 'Asia/Seoul'
```

---

## 8. 전체 흐름 정리

```
① 브라우저가 URL 요청 (예: GET /students/)
   ↓
② URL Resolver (urls.py)
   패턴 매칭 → StudentListView 또는 student_list 함수 결정
   ↓
③ View (views.py)
   request 분석 → Model에 쿼리 요청 → 결과 수집
   ↓
④ Model (models.py) + ORM
   Python 코드를 SQL로 변환 → DB 조회
   ↓⑤
⑤ DB (MySQL/SQLite 등)
   SQL 실행 → 결과 반환
   ↓
⑥ View → Template
   데이터를 context로 Template에 전달
   ↓
⑦ Template (templates/*.html)
   DTL/Jinja로 HTML 렌더링
   ↓
⑧ Response
   HttpResponse/JsonResponse/render 중 선택해 브라우저로 반환
   ↓
⑨ 브라우저
   HTML 렌더링 (Bootstrap/JavaScript 적용)
```

### Django vs 다른 Python 웹 프레임워크

|프레임워크|특징|적합한 경우|
|---|---|---|
|**Django**|풀스택, 배터리 포함, ORM, Admin|중~대규모, 빠른 개발|
|**Flask**|마이크로, 가볍고 유연|소규모, 커스텀 많이 필요|
|**FastAPI**|비동기, 자동 API 문서|ML 모델 API 서버, 고성능 API|
|**Streamlit**|데이터 앱 전용|ML 결과 시각화, 빠른 프로토타입|

ML 프로젝트에서:

- 관리자 페이지 + 데이터 관리 → Django
- 간단한 ML API → FastAPI
- 빠른 시각화 대시보드 → Streamlit
