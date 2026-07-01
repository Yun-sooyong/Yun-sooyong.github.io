---
title: 046 django (Form, View, Model)
tag:
  - 헬스케어 ai
  - django
description: 260624 수업 내용 정리
---

# 장고 (Form : View : Model)

---

## 1. Form : View : Model — 3위 일체

Model, View, Form은 Django에서 **서로 분리되어 있지만 항상 함께 동작**한다. 이 세 가지를 "3위 일체"라고 부르는 이유다.

- **Model**: DB 테이블 구조와 데이터를 정의
- **View**: 요청을 받아 Model에서 데이터를 가져오고 Form을 처리하는 중간 처리자
- **Form**: 사용자 입력을 받고 검증하고 DB에 저장

```
사용자 입력 → Form(검증) → View(처리 흐름) → Model(DB 저장)
DB 데이터   → Model      → View(context)   → Template(출력)
```

**ModelForm**은 Model과 Form을 결합한 것이다. Model이 정의한 필드를 자동으로 Form 필드로 변환해주고, `save()` 한 번으로 DB 저장까지 처리한다.

---

## 2. GET과 POST — "빈폼 vs 속이 찬폼"

Django View에서 가장 많이 만나는 패턴이다. 하나의 URL과 하나의 View 함수가 GET과 POST 두 경우를 모두 처리한다.

```
GET 요청 (브라우저가 페이지 처음 열 때):
  → 빈 폼을 보여준다
  → form = StudentForm()           ← 아무 데이터도 없는 빈 껍데기
  → form.is_bound == False

POST 요청 (사용자가 폼을 작성하고 제출할 때):
  → 속이 찬 폼 (사용자 입력 데이터가 들어 있음)
  → form = StudentForm(request.POST)  ← 입력 데이터가 바인딩된 폼
  → form.is_bound == True
```

이것이 슬라이드의 "Get: 빈폼 / Post: 속이 찬폼"의 의미다.

```python
# views.py — GET/POST 통합 처리 패턴
def student_create(request):
    if request.method == 'POST':
        # POST: 사용자가 제출한 데이터로 폼 생성 (속이 찬폼)
        form = StudentForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect('student-list')
        # is_valid() 실패 시 아래로 내려가 에러 포함한 폼을 다시 보여줌
    else:
        # GET: 빈 폼 생성 (빈폼)
        form = StudentForm()

    # GET이든 POST 검증 실패든 똑같이 폼을 화면에 보여줌
    return render(request, 'students/form.html', {'form': form})
```

---

## 3. Form과 ModelForm

### Forms.Form — 독립적인 폼

Model과 직접 연결되지 않는 일반 폼이다. 로그인, 검색, 문의하기처럼 DB에 저장하지 않는 경우에 사용한다.

```python
# forms.py
from django import forms

class ContactForm(forms.Form):
    name    = forms.CharField(max_length=100, label='이름')
    email   = forms.EmailField(label='이메일')
    message = forms.CharField(widget=forms.Textarea, label='내용')
    score   = forms.IntegerField(min_value=0, max_value=100)
```

### Forms.ModelForm — Model 기반 폼

Model의 필드를 자동으로 Form 필드로 변환한다. `save()` 한 번으로 DB까지 저장된다. CRUD의 대부분은 이 방식으로 구현한다.

```python
# forms.py
from django import forms
from .models import Student

class StudentForm(forms.ModelForm):
    class Meta:
        model  = Student
        fields = '__all__'         # 모든 필드 사용
        # fields = ['name', 'kor', 'mat']  # 일부만 지정
        # exclude = ['bunho']              # 특정 필드 제외
        widgets = {
            'name': forms.TextInput(attrs={
                'class': 'form-control',
                'placeholder': '이름을 입력하세요'
            }),
            'kor': forms.NumberInput(attrs={
                'class': 'form-control', 'min': 0, 'max': 100
            }),
        }
        labels = {
            'name': '이름', 'kor': '국어', 'mat': '수학', 'eng': '영어'
        }
        error_messages = {
            'name': {'required': '이름은 필수 항목입니다.'}
        }
```

### Form vs ModelForm 비교

|구분|Forms.Form|Forms.ModelForm|
|---|---|---|
|Model 연결|없음|있음 (자동 필드 생성)|
|DB 저장|직접 구현 필요|`form.save()` 한 줄|
|필드 정의|모두 직접 작성|`Meta.fields`로 지정|
|사용 사례|로그인, 검색, 이메일 발송|CRUD 데이터 처리|

---

## 4. 폼 상태 확인과 검증

### Form.is_bound — 폼에 데이터가 바인딩됐는가

```python
form_empty = StudentForm()                 # GET → 빈폼
form_bound = StudentForm(request.POST)     # POST → 속이 찬폼

print(form_empty.is_bound)   # False
print(form_bound.is_bound)   # True
```

`is_bound`가 False인 폼은 유효성 검사를 할 수 없다. `is_valid()`를 호출해도 항상 False를 반환한다.

### Form.is_valid() — 유효성 검사 통과 여부

폼에 바인딩된 데이터가 각 필드의 규칙(필수값, 타입, 길이, 범위 등)을 모두 통과하면 True를 반환한다. 내부적으로 `clean()` 메서드 체계를 순서대로 호출한다.

```python
form = StudentForm(request.POST)

if form.is_valid():
    # 검증 통과 → cleaned_data에 Python 타입으로 변환된 데이터
    print(form.cleaned_data)
    # {'name': '홍길동', 'kor': 90, 'mat': 85, 'eng': 78}
    form.save()   # DB 저장
else:
    # 검증 실패 → errors에 오류 내용
    print(form.errors)
    # {'kor': ['이 값은 100 이하여야 합니다.']}
```

### Form.clean() — 커스텀 유효성 검사

Django 기본 검증 외에 추가적인 비즈니스 규칙을 직접 정의할 때 사용한다.

`clean_필드명()` 메서드로 개별 필드 검증을, `clean()` 메서드로 여러 필드를 함께 검증한다.

```python
class StudentForm(forms.ModelForm):
    class Meta:
        model  = Student
        fields = '__all__'

    # 개별 필드 검증: clean_<필드명>()
    def clean_kor(self):
        value = self.cleaned_data.get('kor')
        if value is not None and not (0 <= value <= 100):
            raise forms.ValidationError('국어 점수는 0~100 사이여야 합니다.')
        return value   # 반드시 값을 반환해야 함

    def clean_name(self):
        name = self.cleaned_data.get('name', '').strip()
        if len(name) < 2:
            raise forms.ValidationError('이름은 2자 이상이어야 합니다.')
        return name

    # 여러 필드를 함께 검증: clean()
    def clean(self):
        cleaned = super().clean()
        kor = cleaned.get('kor', 0)
        mat = cleaned.get('mat', 0)
        eng = cleaned.get('eng', 0)
        if (kor + mat + eng) / 3 < 10:
            raise forms.ValidationError('평균 점수가 너무 낮습니다. 데이터를 확인하세요.')
        return cleaned
```

`is_valid()` 호출 시 내부 실행 순서:

```
is_valid() 호출
   ↓
각 필드의 기본 검증 (필수값, 타입, 길이 등)
   ↓
clean_<필드명>() 개별 필드 커스텀 검증 (필드마다)
   ↓
clean() 전체 폼 커스텀 검증
   ↓
모두 통과 → True, cleaned_data에 데이터 저장
오류 발생 → False, errors에 오류 내용 저장
```

---

## 5. 폼을 HTML로 렌더링 — form.as_*

슬라이드 오른쪽의 "HTML 형식으로 변환" 박스가 이 세 가지다.

|메서드|렌더링 결과|특징|
|---|---|---|
|`{{ form.as_table }}`|`<tr><th>라벨</th><td>입력필드</td></tr>`|표 형식|
|`{{ form.as_p }}`|`<p>라벨: 입력필드</p>`|단락 형식, 가장 단순|
|`{{ form.as_ul }}`|`<li>라벨: 입력필드</li>`|목록 형식|

```html
<!-- templates/students/form.html -->
<form method="post">
    {% csrf_token %}   <!-- CSRF 보안 토큰 (POST 폼에 필수) -->

    <!-- 방법 1: 자동 렌더링 (빠른 구현) -->
    {{ form.as_p }}

    <!-- 방법 2: Bootstrap 스타일을 위한 수동 렌더링 (실무) -->
    {% for field in form %}
    <div class="mb-3">
        <label for="{{ field.id_for_label }}" class="form-label">
            {{ field.label }}
            {% if field.field.required %}<span class="text-danger">*</span>{% endif %}
        </label>
        {{ field }}   <!-- 실제 입력 위젯 -->
        {% if field.errors %}
            <div class="text-danger small">{{ field.errors }}</div>
        {% endif %}
        {% if field.help_text %}
            <small class="text-muted">{{ field.help_text }}</small>
        {% endif %}
    </div>
    {% endfor %}

    <button type="submit" class="btn btn-primary">저장</button>
    <a href="{% url 'student-list' %}" class="btn btn-secondary">취소</a>
</form>
```

---

## 6. FormSet, ModelFormSet, InlineFormSet

하나의 폼이 아니라 **같은 모델의 여러 인스턴스를 한 페이지에서 동시에 처리**할 때 사용한다.

### FormSet — 같은 폼을 여러 개 묶기

```python
from django.forms import formset_factory
from .forms import StudentForm

StudentFormSet = formset_factory(StudentForm, extra=3)   # 빈 폼 3개

def student_bulk_create(request):
    if request.method == 'POST':
        formset = StudentFormSet(request.POST)
        if formset.is_valid():
            for form in formset:
                if form.cleaned_data:   # 실제로 입력된 폼만 처리
                    form.save()
            return redirect('student-list')
    else:
        formset = StudentFormSet()
    return render(request, 'students/formset.html', {'formset': formset})
```

```html
<!-- templates/students/formset.html -->
<form method="post">
    {% csrf_token %}
    {{ formset.management_form }}   <!-- FormSet에 필수 -->
    {% for form in formset %}
        {{ form.as_p }}
        <hr>
    {% endfor %}
    <button type="submit">저장</button>
</form>
```

### ModelFormSet — DB 모델과 연동된 FormSet

기존 DB 데이터를 여러 행 동시에 편집할 때 유용하다.

```python
from django.forms import modelformset_factory
from .models import Student

StudentModelFormSet = modelformset_factory(
    Student,
    fields=['name', 'kor', 'mat', 'eng'],
    extra=2,          # 기존 데이터 + 빈 폼 2개
    can_delete=True   # 삭제 체크박스 추가
)

def student_bulk_edit(request):
    if request.method == 'POST':
        formset = StudentModelFormSet(request.POST)
        if formset.is_valid():
            formset.save()   # 수정 + 삭제 + 신규 모두 한 번에 처리
            return redirect('student-list')
    else:
        formset = StudentModelFormSet(queryset=Student.objects.all())
    return render(request, 'students/bulk_edit.html', {'formset': formset})
```

### InlineFormSet — 부모-자식(ForeignKey) 관계 처리

슬라이드의 "DB 모델과의 연동 여부와 부모-자식(ForeignKey) 관계성 표현 여부"가 이것이다. 학급(Classroom)을 수정할 때 그 학급 소속 학생(Student)들을 한 페이지에서 함께 편집하는 경우다.

```python
# models.py
class Classroom(models.Model):
    name = models.CharField(max_length=50)

class Student(models.Model):
    classroom = models.ForeignKey(Classroom, on_delete=models.CASCADE)  # 부모-자식
    name      = models.CharField(max_length=50)
    kor       = models.SmallIntegerField(default=0)
```

```python
# forms.py
from django.forms import inlineformset_factory

# Classroom(부모) 하나에 속한 Student(자식) 여러 개를 한 페이지에서 편집
StudentInlineFormSet = inlineformset_factory(
    Classroom, Student,       # 부모 모델, 자식 모델
    fields=['name', 'kor'],
    extra=3,
    can_delete=True
)

# views.py
def classroom_edit(request, pk):
    classroom = get_object_or_404(Classroom, pk=pk)
    if request.method == 'POST':
        formset = StudentInlineFormSet(request.POST, instance=classroom)
        if formset.is_valid():
            formset.save()
            return redirect('classroom-list')
    else:
        # instance=classroom: 이 학급에 속한 학생들만 자동으로 로드
        formset = StudentInlineFormSet(instance=classroom)
    return render(request, 'classrooms/edit.html',
                  {'formset': formset, 'classroom': classroom})
```

|구분|FormSet|ModelFormSet|InlineFormSet|
|---|---|---|---|
|DB 모델 연동|없음|있음|있음|
|관계 표현|없음|없음|ForeignKey 부모-자식|
|삭제 지원|`can_delete=True`|`can_delete=True`|`can_delete=True`|
|사용 사례|비모델 폼 묶음|동일 모델 여러 행 편집|부모 수정 시 자식 함께 편집|

---

## 7. URL 매개변수 처리와 name → URL 처리

슬라이드의 "url 매개변수 처리방법(class, function)"과 "name → url 처리 방법(view, template)"이다.

### URL 매개변수 처리 — Function 방식 (FBV)

```python
# urls.py
path('students/<int:pk>/',     views.student_detail,   name='student-detail'),
path('students/<str:name>/',   views.student_by_name,  name='student-by-name'),

# views.py
def student_detail(request, pk):       # URL 매개변수가 함수 인자로 바로 들어옴
    student = get_object_or_404(Student, pk=pk)
    return render(request, 'students/detail.html', {'student': student})
```

### URL 매개변수 처리 — Class 방식 (CBV)

```python
# views.py
class StudentDetailView(DetailView):
    model = Student

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        pk = self.kwargs['pk']   # self.kwargs로 URL 매개변수 접근
        context['extra'] = f'pk={pk}'
        return context
```

### name → URL 처리 — 뷰에서

URL에 이름을 붙이면 하드코딩 없이 URL을 동적으로 생성할 수 있다. URL 구조가 바뀌어도 `name`만 유지되면 뷰와 템플릿을 수정하지 않아도 된다.

```python
# urls.py — name 부여
path('students/',                   StudentListView.as_view(),   name='student-list'),
path('students/<int:pk>/',          StudentDetailView.as_view(), name='student-detail'),
path('students/new/',               StudentCreateView.as_view(), name='student-create'),

# views.py — 뷰에서 name으로 이동
from django.urls import reverse, reverse_lazy
from django.shortcuts import redirect

def some_view(request):
    return redirect('student-list')                                # 이름으로 리다이렉트
    return redirect(reverse('student-detail', kwargs={'pk': 1}))  # 매개변수 포함

# CBV에서는 reverse_lazy 사용 (클래스 정의 시점에 URL 아직 로드 안 됨)
class StudentCreateView(CreateView):
    success_url = reverse_lazy('student-list')
```

### name → URL 처리 — 템플릿에서

```html
<a href="{% url 'student-list' %}">전체 목록</a>
<a href="{% url 'student-detail' pk=student.pk %}">{{ student.name }}</a>
<a href="{% url 'student-create' %}">새 학생 추가</a>

<!-- 폼 action에도 사용 -->
<form method="post" action="{% url 'student-create' %}">
    {% csrf_token %}
    ...
</form>
```

---

## 8. Models → Objects Manager → QuerySet → View → Template 흐름

슬라이드 하단 노란색 박스의 전체 데이터 처리 파이프라인이다.

```
Models Class → Objects Manager → QuerySet(쿼리 가능) → View(Contexts) → Template(Form class)
```

### ① Models Class

`models.py`에 정의한 Python 클래스. DB 테이블 구조를 정의한다.

```python
class Student(models.Model):
    name = models.CharField(max_length=50)
    kor  = models.SmallIntegerField(default=0)
```

### ② Objects Manager

모든 Model에 자동으로 붙는 `objects` 속성이다. DB에 쿼리를 보내는 진입점 역할을 한다.

```python
Student.objects          # Manager 객체
Student.objects.all()    # 전체 조회 → QuerySet 반환
Student.objects.filter() # 조건 조회 → QuerySet 반환
Student.objects.get()    # 단일 조회 → Model 인스턴스 반환 (없으면 예외)
Student.objects.create() # 생성 후 인스턴스 반환
```

### ③ QuerySet (쿼리 가능)

Manager가 반환하는 데이터 컬렉션이다. **지연 평가(Lazy Evaluation)** 가 핵심 특성이다. QuerySet을 정의해도 실제 SQL은 데이터가 실제로 필요한 시점까지 실행되지 않는다.

```python
# 이 시점에는 DB 쿼리 미실행 (QuerySet 정의만)
qs = Student.objects.filter(kor__gte=80).order_by('-kor')

# 아래 시점에 SELECT 쿼리가 DB로 전달됨
for s in qs:     # 반복할 때
list(qs)         # 리스트로 변환할 때
qs[0]            # 인덱싱/슬라이싱할 때
len(qs)          # 길이를 구할 때

# QuerySet 체이닝 — 여러 조건을 연결
Student.objects.filter(kor__gte=80).exclude(name='홍길동').order_by('name')[:10]
```

### ④ View — Contexts

QuerySet을 받아서 Template에 넘길 딕셔너리(context)를 만든다.

```python
def student_list(request):
    context = {
        'students':    Student.objects.all().order_by('-kor'),
        'top_student': Student.objects.order_by('-kor').first(),
        'count':       Student.objects.count(),
        'form':        StudentForm(),   # 폼도 함께 전달 가능
    }
    return render(request, 'students/list.html', context)
```

### ⑤ Template — Form class

context 데이터를 출력하고, 필요하면 Form을 함께 렌더링한다.

```html
<p>총 {{ count }}명 / 최고점: {{ top_student.name }} ({{ top_student.kor }}점)</p>

{% for s in students %}
<tr>
    <td>{{ s.name }}</td>
    <td>{{ s.kor }}</td>
</tr>
{% endfor %}

<!-- 폼도 같은 페이지에 표시 가능 -->
{{ form.as_p }}
```

---

## 9. General View (CBV) — 제네릭 뷰 전체 정리

슬라이드 왼쪽의 General View 목록이다. 반복적인 CRUD 패턴을 상속 하나로 해결한다.

```python
from django.views.generic import (
    CreateView, ListView, DetailView,
    UpdateView, DeleteView, FormView
)
from django.urls import reverse_lazy
from .models import Student
from .forms import StudentForm


class StudentListView(ListView):
    """목록 조회 — GET /students/"""
    model               = Student
    template_name       = 'students/list.html'
    context_object_name = 'students'   # 기본값은 'object_list'
    ordering            = ['-kor']
    paginate_by         = 10

    def get_queryset(self):
        """검색 파라미터 처리 등 커스텀 필터링"""
        qs   = super().get_queryset()
        name = self.request.GET.get('name')
        return qs.filter(name__icontains=name) if name else qs

    def get_context_data(self, **kwargs):
        """추가 context 데이터 전달"""
        context = super().get_context_data(**kwargs)
        context['total_count'] = Student.objects.count()
        return context


class StudentDetailView(DetailView):
    """단일 조회 — GET /students/<pk>/"""
    model         = Student
    template_name = 'students/detail.html'
    # URL의 pk를 자동으로 처리해 해당 객체를 context에 담아 넘김


class StudentCreateView(CreateView):
    """생성 — GET: 빈폼 / POST: 저장"""
    model         = Student
    form_class    = StudentForm
    template_name = 'students/form.html'
    success_url   = reverse_lazy('student-list')

    def form_valid(self, form):
        """저장 성공 시 추가 처리 (예: 로그인 사용자 자동 설정)"""
        # form.instance.created_by = self.request.user
        return super().form_valid(form)


class StudentUpdateView(UpdateView):
    """수정 — GET: 기존 데이터 채운 폼 / POST: 저장"""
    model         = Student
    form_class    = StudentForm
    template_name = 'students/form.html'
    success_url   = reverse_lazy('student-list')


class StudentDeleteView(DeleteView):
    """삭제 — GET: 확인 페이지 / POST: 삭제 실행"""
    model         = Student
    template_name = 'students/confirm_delete.html'
    success_url   = reverse_lazy('student-list')


class StudentFormView(FormView):
    """Model과 무관한 폼 처리 — 예: 검색, 이메일 발송"""
    form_class    = ContactForm
    template_name = 'contact.html'
    success_url   = reverse_lazy('home')

    def form_valid(self, form):
        data = form.cleaned_data
        # send_mail(data['email'], data['message'])
        return super().form_valid(form)
```

```python
# urls.py — CBV 연결
urlpatterns = [
    path('students/',                   StudentListView.as_view(),   name='student-list'),
    path('students/<int:pk>/',          StudentDetailView.as_view(), name='student-detail'),
    path('students/new/',               StudentCreateView.as_view(), name='student-create'),
    path('students/<int:pk>/edit/',     StudentUpdateView.as_view(), name='student-update'),
    path('students/<int:pk>/delete/',   StudentDeleteView.as_view(), name='student-delete'),
    path('contact/',                    StudentFormView.as_view(),   name='contact'),
]
```

### General View 비교표

|뷰 클래스|HTTP|역할|기본 템플릿 이름|
|---|---|---|---|
|`ListView`|GET|목록 조회|`<모델>_list.html`|
|`DetailView`|GET|단일 조회|`<모델>_detail.html`|
|`CreateView`|GET/POST|생성|`<모델>_form.html`|
|`UpdateView`|GET/POST|수정|`<모델>_form.html`|
|`DeleteView`|GET/POST|삭제|`<모델>_confirm_delete.html`|
|`FormView`|GET/POST|일반 폼 처리|직접 지정|

---

## 10. QuerySet, Form, View 구성 요소 정리

슬라이드 왼쪽 하단 회색 박스다. 세 요소가 하나의 CRUD 기능에서 각자 역할을 맡는다.

|요소|역할|주요 위치|
|---|---|---|
|**QuerySet**|DB에서 가져온 데이터 컬렉션. View에서 context로 Template에 전달|`objects.filter()`, `objects.all()`|
|**Form**|사용자 입력 처리 + 검증 + DB 저장. View가 생성, Template이 렌더링|`forms.py`, `form.is_valid()`, `form.save()`|
|**View**|QuerySet과 Form을 연결하는 중간 처리자|`views.py`, `render()`, `redirect()`|

```
전체 CRUD 흐름:

[목록]   QuerySet → context → Template 출력
[생성]   GET: 빈 Form → Template / POST: Form 검증 → Model 저장
[수정]   GET: 기존 데이터 채운 Form → Template / POST: Form 검증 → Model 저장
[삭제]   GET: 확인 페이지 → Template / POST: Model 삭제
```

---

## 11. 데이터 흐름도

### 전체 요청-응답 흐름

Django에서 브라우저 요청이 들어와 응답이 나가기까지 데이터가 어떤 경로로 이동하는지 단계별로 정리한다.

```
브라우저
  │
  │  HTTP Request (GET or POST)
  ▼
urls.py (URL Resolver)
  │  패턴 매칭 → 어떤 View를 실행할지 결정
  │  예: /students/3/ → StudentDetailView, pk=3
  ▼
views.py (View)
  │
  ├─ GET 요청인 경우 ─────────────────────────────────────
  │    ① Model.objects.xxx() 로 QuerySet 생성
  │    ② context = {'key': QuerySet or 객체} 딕셔너리 구성
  │    ③ render(request, 'template.html', context)
  │
  ├─ POST 요청인 경우 ─────────────────────────────────────
  │    ① form = MyForm(request.POST) 로 바인딩
  │    ② form.is_valid() → clean_xxx() → clean() 순서로 검증
  │    ③ 통과: form.save() → DB 저장 → redirect()
  │       실패: form.errors 포함해서 render() 로 다시 렌더링
  │
  ▼
models.py (Model + ORM)
  │  Python 코드 → SQL 변환 → DB 조회/저장
  │  예: Student.objects.filter(kor__gte=80)
  │      → SELECT * FROM student WHERE kor >= 80
  ▼
Database (MySQL / SQLite 등)
  │  SQL 실행 → 결과 반환
  ▼
views.py (View)
  │  결과를 context에 담아 Template으로 전달
  ▼
templates/xxx.html (Template)
  │  DTL/Jinja로 context 데이터를 HTML에 삽입
  │  {{ 변수 }}, {% for %}, {% if %}, {{ form.as_p }} 등
  ▼
브라우저
   HTTP Response (HTML / JSON / Redirect)
```

---

### CRUD별 데이터 흐름 상세

#### 목록 조회 (ListView — GET)

```
브라우저 GET /students/
  ↓
urls.py: StudentListView.as_view() 호출
  ↓
StudentListView.get_queryset()
  → Student.objects.all().order_by('-kor')   ← QuerySet 생성 (아직 DB 미조회)
  ↓
get_context_data()
  → context = {'students': QuerySet, 'count': 100}
  ↓
Template 렌더링 시 QuerySet 실제 평가
  → SELECT * FROM student ORDER BY kor DESC   ← 이 시점에 DB 조회
  ↓
HTML 생성 → 브라우저에 응답
```

#### 생성 (CreateView — GET/POST)

```
─── GET 요청: 빈 폼 보여주기 ────────────────────────
브라우저 GET /students/new/
  ↓
StudentCreateView → form = StudentForm()   ← 빈폼 (is_bound=False)
  ↓
context = {'form': form}
  ↓
Template: {{ form.as_p }}  →  빈 입력 필드 HTML 생성
  ↓
브라우저에 빈 폼 출력

─── POST 요청: 입력 데이터 처리 ─────────────────────
브라우저 POST /students/new/ (폼 데이터 전송)
  ↓
StudentCreateView → form = StudentForm(request.POST)   ← 바인딩 (is_bound=True)
  ↓
form.is_valid()
  │
  ├─ True (검증 통과)
  │    form.cleaned_data = {'name': '홍길동', 'kor': 90, ...}
  │    form.save()
  │      → INSERT INTO student (name, kor, ...) VALUES ('홍길동', 90, ...)
  │    redirect('student-list')   → 브라우저 목록 페이지로 이동
  │
  └─ False (검증 실패)
       form.errors = {'kor': ['이 값은 100 이하여야 합니다.']}
       render() → 에러 메시지 포함한 폼 다시 출력
```

#### 수정 (UpdateView — GET/POST)

```
─── GET 요청: 기존 데이터 채운 폼 보여주기 ───────────
브라우저 GET /students/3/edit/
  ↓
urls.py: pk=3 추출
  ↓
Student.objects.get(pk=3)   → SELECT * FROM student WHERE id=3
  ↓
form = StudentForm(instance=student)   ← 기존 데이터가 채워진 폼
  ↓
Template: {{ form.as_p }}  →  기존 값이 들어 있는 입력 필드 출력

─── POST 요청: 수정 데이터 처리 ─────────────────────
브라우저 POST /students/3/edit/ (수정 데이터 전송)
  ↓
form = StudentForm(request.POST, instance=student)   ← 기존 인스턴스 + 새 데이터
  ↓
form.is_valid() → form.save()
  → UPDATE student SET kor=95 WHERE id=3
  ↓
redirect('student-list')
```

#### 삭제 (DeleteView — GET/POST)

```
─── GET 요청: 삭제 확인 페이지 ──────────────────────
브라우저 GET /students/3/delete/
  ↓
Student.objects.get(pk=3)   → SELECT (확인용)
  ↓
Template: "정말 삭제하시겠습니까?" 확인 페이지 출력

─── POST 요청: 삭제 실행 ─────────────────────────────
브라우저 POST /students/3/delete/
  ↓
student.delete()
  → DELETE FROM student WHERE id=3
  ↓
redirect('student-list')
```

---

### Form 데이터 흐름 상세

```
사용자 입력 (브라우저 HTML 폼)
  ↓ POST 전송
request.POST = {'name': '홍길동', 'kor': '90', 'mat': '85', 'eng': '78'}
  (모든 값은 문자열로 들어옴)
  ↓
form = StudentForm(request.POST)
  ↓
form.is_valid()
  ┌─────────────────────────────────────────────┐
  │ 각 필드 기본 검증                            │
  │   CharField:       문자열 → 그대로           │
  │   SmallIntegerField: '90' → int 90 변환     │
  │   EmailField:      이메일 형식 확인           │
  │   필수 여부 확인                             │
  ├─────────────────────────────────────────────┤
  │ clean_kor() 개별 커스텀 검증                 │
  │   0 <= value <= 100 확인                    │
  ├─────────────────────────────────────────────┤
  │ clean() 전체 커스텀 검증                     │
  │   평균 점수 확인 등                          │
  └─────────────────────────────────────────────┘
  ↓
검증 통과
  form.cleaned_data = {
      'name': '홍길동',   ← 문자열 그대로
      'kor':  90,          ← int로 변환됨
      'mat':  85,
      'eng':  78
  }
  ↓
form.save()
  → Student 인스턴스 생성
  → INSERT INTO student (name, kor, mat, eng) VALUES ('홍길동', 90, 85, 78)
  ↓
DB 저장 완료 → redirect
```

---

### ModelForm ↔ Model ↔ DB 흐름

```
models.py                forms.py               DB
─────────                ────────               ──
class Student            class StudentForm      table: student
  name: CharField    ←→    name: TextInput    ↔  name: VARCHAR
  kor:  SmallInt     ←→    kor:  NumberInput  ↔  kor:  TINYINT
  mat:  SmallInt     ←→    mat:  NumberInput  ↔  mat:  TINYINT
  eng:  SmallInt     ←→    eng:  NumberInput  ↔  eng:  TINYINT

ModelForm이 Model 필드를 읽어서 Form 필드를 자동 생성
Model이 ORM을 통해 DB 컬럼과 매핑

저장 경로:
  request.POST
    → Form(유효성 검사)
      → ModelForm.save()
        → Model 인스턴스
          → ORM(Python → SQL)
            → DB INSERT/UPDATE
```

---

### Admin ↔ Model ↔ DB 흐름

```
admin.py
  admin.site.register(Student)
        ↓
/admin URL 접속
        ↓
Django Admin이 Student 모델을 읽어서
  자동으로 목록/추가/수정/삭제 페이지 생성
        ↓
관리자가 웹에서 데이터 조작
        ↓
Student.objects.create() / .save() / .delete()
        ↓
DB (MySQL/SQLite)

흐름:
  Admin → ModelForm(자동 생성) → Model → ORM → DB
```

---

### QuerySet의 지연 평가 흐름

```
코드 실행 순서              실제 SQL 실행 시점
──────────────              ──────────────────

① qs = Student.objects.all()           → SQL 없음
② qs = qs.filter(kor__gte=80)          → SQL 없음
③ qs = qs.order_by('-kor')             → SQL 없음
④ qs = qs[:10]                         → SQL 없음

⑤ context = {'students': qs}           → SQL 없음
⑥ return render(request, ..., context) → SQL 없음

⑦ Template에서 {% for s in students %} → SQL 실행!
   SELECT * FROM student
   WHERE kor >= 80
   ORDER BY kor DESC
   LIMIT 10;

→ 실제로 데이터가 필요한 순간(Template 렌더링)에만 DB 조회
→ 조건을 계속 붙여도 쿼리가 누적되지 않음 (효율적)
```

---

### 전체 흐름 한눈에 요약

```
브라우저
   │ GET /students/new/
   ▼
urls.py ──→ StudentCreateView
               │
          [GET] form = StudentForm()  ──→ Template ──→ 빈폼 HTML ──→ 브라우저
               │
          [POST] form = StudentForm(request.POST)
               │
               ├─ is_valid() True
               │     cleaned_data → form.save()
               │         Model → ORM → DB INSERT
               │     redirect('student-list')
               │         ↓
               │     브라우저 → GET /students/
               │         ↓
               │     QuerySet → context → Template → HTML
               │
               └─ is_valid() False
                     form.errors → render() → 에러 포함 폼 HTML → 브라우저
```