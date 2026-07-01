---
title: 047 Authorization(권한) & Authentication(인증)
tag:
  - 헬스케어 ai
  - django
description: 260625 수업 내용 정리
---

# Django — Authorization(권한) & Authentication(인증)

---

## 1. 인증과 권한의 차이

웹 서비스를 만들 때 두 가지 질문을 항상 구분해서 생각해야 한다.

**Authentication(인증)** 은 "당신이 누구인가?"에 대한 답을 확인하는 과정이다. 로그인할 때 아이디와 비밀번호를 입력하면 Django가 DB에서 해당 사용자를 찾고 비밀번호가 맞는지 검증한다. 이 과정을 통과하면 "이 사람은 홍길동이다"라는 것이 확인된 것이다. 결과는 `request.user.is_authenticated`로 True 또는 False가 된다.

**Authorization(권한)** 은 "당신이 무엇을 할 수 있는가?"를 결정하는 과정이다. 로그인에 성공했다고 해서 모든 기능을 쓸 수 있는 것은 아니다. 일반 사용자는 자신의 글만 수정할 수 있고, 관리자는 모든 글을 수정할 수 있다. 이런 제한이 권한이다.

```
현실 세계 비유:

Authentication(인증):
  회사 건물에 들어가기 위해 사원증을 찍는다
  → "당신이 우리 직원임을 확인했다"

Authorization(권한):
  사원증으로 들어간 후, 연구소 구역은 연구원만 출입 가능
  → "당신이 이 구역에 들어갈 자격이 있는가?"

→ 인증이 먼저, 권한 확인이 그 다음
```

Django는 `django.contrib.auth` 패키지 하나로 두 가지를 모두 처리한다. `settings.py`의 `INSTALLED_APPS`에 `django.contrib.auth`가 기본으로 포함되어 있다.

---

## 2. Django 권한 시스템 구조 — 관리 Table 3종

슬라이드 중앙의 관리 Table이다. Django가 인증/권한을 위해 `migrate` 실행 시 자동으로 생성하는 DB 테이블들이다.

```
관리 Table
  ├─ auth_user         (사용자 계정 정보 + 개인별 permission)
  ├─ auth_group        (권한 묶음, User + permission의 조합)
  └─ auth_permission   (실제 권한 항목 목록)
```

이 세 테이블과 함께 관계 테이블들(`auth_user_groups`, `auth_user_user_permissions`, `auth_group_permissions`)이 함께 생성된다. 관리자 페이지(`/admin`)의 "인증 및 권한" 섹션에서 이 테이블들을 시각적으로 관리할 수 있다.

### user 테이블 — 개인별 permission

`django.contrib.auth.models.User` 모델이 이 테이블과 매핑된다. 사용자 계정 정보와 함께 개인에게 직접 부여된 권한(`user_permissions`)과 소속 그룹(`groups`)을 관리한다.

|필드|타입|설명|
|---|---|---|
|`id`|AutoField|기본키|
|`username`|CharField(150)|로그인 아이디. 고유값|
|`password`|CharField|해시된 비밀번호. 평문 저장 절대 불가|
|`email`|EmailField|이메일 주소|
|`first_name`|CharField(150)|이름|
|`last_name`|CharField(150)|성|
|`is_active`|BooleanField|False면 로그인 자체가 불가 (계정 비활성화)|
|`is_staff`|BooleanField|True면 `/admin` 페이지에 로그인 가능|
|`is_superuser`|BooleanField|True면 모든 권한을 자동으로 보유|
|`last_login`|DateTimeField|마지막 로그인 시각|
|`date_joined`|DateTimeField|계정 생성 시각|
|`groups`|ManyToManyField|소속 그룹들|
|`user_permissions`|ManyToManyField|개인에게 직접 부여된 권한들|

`is_active`, `is_staff`, `is_superuser` 세 플래그의 차이를 명확히 이해해야 한다.

```
is_active = False:
  로그인 자체가 불가능함
  계정을 물리적으로 삭제하지 않고 일시 정지할 때 사용
  ex: 이용 약관 위반 사용자 계정 정지

is_staff = True:
  /admin 페이지에 접속할 수 있음
  하지만 어떤 모델을 볼 수 있는지는 개별 권한으로 제어
  ex: 고객센터 직원 → 주문 테이블만 볼 수 있게 설정

is_superuser = True:
  모든 권한을 자동으로 가짐. 따로 권한 부여할 필요 없음
  has_perm()이 항상 True를 반환
  ex: 시스템 관리자
  
  주의: 운영 환경에서 superuser 계정은 최소한으로 유지해야 함
```

### group 테이블 — User와 Permission의 조합

그룹은 **여러 사용자에게 같은 권한 묶음을 한 번에 부여**하기 위한 개념이다. 슬라이드 관리자 화면의 "그룹" 항목이 이것이다.

그룹 없이 권한을 관리하면 다음 문제가 생긴다.

```
그룹 없이 관리하는 경우:
  편집자 100명에게 각각 polls.add_choice, polls.change_choice 권한을 부여
  나중에 polls.view_choice 권한도 추가해야 하면?
  → 100명을 한 명씩 수정해야 함 → 매우 비효율적

그룹으로 관리하는 경우:
  "편집자" 그룹에 polls.add_choice, polls.change_choice 권한을 부여
  100명을 모두 이 그룹에 소속
  나중에 권한 추가가 필요하면?
  → 그룹 하나만 수정 → 100명 전체에 즉시 반영
```

실제로는 이런 식으로 그룹을 설계한다.

```
그룹 예시 설계:

그룹: "읽기 전용"
  → *.view_* 권한들만 보유
  → 모든 데이터를 조회는 가능하지만 수정 불가

그룹: "편집자"
  → *.view_*, *.add_*, *.change_* 권한
  → 조회, 추가, 수정은 가능하지만 삭제 불가

그룹: "관리자"
  → 모든 권한
  → 조회, 추가, 수정, 삭제 모두 가능

사용자 배정:
  일반 직원 → "읽기 전용" 그룹
  콘텐츠 담당자 → "편집자" 그룹
  팀장 → "관리자" 그룹
```

한 사용자는 여러 그룹에 동시에 소속될 수 있고, 최종 권한은 개인 권한과 소속된 모든 그룹의 권한을 합친 것이 된다.

### permission 테이블 — 실제 권한 항목

슬라이드의 검은 박스에 나열된 "Can add choice, Can change choice..." 항목들이 이 테이블의 레코드들이다.

Django는 모델을 `INSTALLED_APPS`에 등록하고 `migrate`를 실행하면 **각 모델마다 자동으로 4가지 기본 권한을 생성**한다. 개발자가 직접 만들 필요가 없다.

```
자동 생성 규칙:
  앱 이름: polls
  모델 이름: Choice

  생성되는 권한:
    name:     "Can add choice"     codename: "add_choice"
    name:     "Can change choice"  codename: "change_choice"
    name:     "Can delete choice"  codename: "delete_choice"
    name:     "Can view choice"    codename: "view_choice"

  권한 참조 형식 (코드에서 사용):
    "polls.add_choice"       → polls 앱의 add_choice 권한
    "polls.delete_question"  → polls 앱의 delete_question 권한
```

뷰에서 권한을 확인할 때 쓰는 형식이 `"앱이름.동사_모델명소문자"`다. 이 규칙을 이해하면 어떤 모델의 어떤 동작에 대한 권한인지 코드만 보고 바로 알 수 있다.

```python
# 커스텀 권한을 추가하는 방법 (선택 사항)
class Student(models.Model):
    name = models.CharField(max_length=50)

    class Meta:
        permissions = [
            ('export_student', '성적 데이터 내보내기'),  # 커스텀 권한
            ('import_student', '성적 데이터 가져오기'),
        ]
# → migrate 후 'sungjukcall.export_student' 형태로 사용 가능
```

### 그룹 권한과 개인 권한 — 코드로 조작하기

슬라이드의 "그룹 권한(groups)과 개인권한(user_permissions)"을 코드로 표현하면 다음과 같다.

```python
from django.contrib.auth.models import User, Permission, Group
from django.contrib.contenttypes.models import ContentType
from .models import Student

# ── 개인 권한 (user_permissions) ──────────────────────
user = User.objects.get(username='홍길동')

# 권한 객체 가져오기
# 방법 1: codename으로 직접 조회
perm_add = Permission.objects.get(codename='add_student')

# 방법 2: ContentType을 통해 조회 (앱과 모델을 명확히 지정)
ct = ContentType.objects.get_for_model(Student)
perm_add    = Permission.objects.get(content_type=ct, codename='add_student')
perm_change = Permission.objects.get(content_type=ct, codename='change_student')
perm_delete = Permission.objects.get(content_type=ct, codename='delete_student')

# 권한 부여
user.user_permissions.add(perm_add)
user.user_permissions.add(perm_add, perm_change)   # 여러 개 동시에
user.user_permissions.set([perm_add, perm_change]) # 기존 권한 대체

# 권한 제거
user.user_permissions.remove(perm_delete)
user.user_permissions.clear()    # 모든 개인 권한 제거

# 권한 확인
user.has_perm('sungjukcall.add_student')    # True or False
user.has_perms(['sungjukcall.add_student',
                'sungjukcall.change_student'])  # 여러 권한 모두 보유 여부

# 보유한 모든 권한 조회
print(user.get_all_permissions())   # 개인 권한 + 그룹 권한 모두
print(user.get_user_permissions())  # 개인 권한만

# ── 그룹 권한 (groups) ────────────────────────────────
# 그룹 생성
editor_group = Group.objects.create(name='편집자')

# 그룹에 권한 부여
editor_group.permissions.add(perm_add, perm_change)
editor_group.permissions.set(
    Permission.objects.filter(content_type__app_label='sungjukcall')
)  # 앱 전체 권한을 그룹에 부여

# 사용자를 그룹에 추가
user.groups.add(editor_group)
user.groups.set([editor_group])   # 기존 그룹 대체

# 그룹 제거
user.groups.remove(editor_group)
user.groups.clear()  # 모든 그룹에서 제거

# 그룹 소속 여부 확인
user.groups.filter(name='편집자').exists()   # True or False

# 중요: 권한 캐시 문제
# 같은 요청 내에서 권한을 부여하고 바로 has_perm()을 확인하면
# 캐시된 이전 결과가 나올 수 있음
# 해결: user 객체를 DB에서 다시 가져오기
from django.contrib.auth.models import User
user = User.objects.get(pk=user.pk)  # 캐시 초기화
user.has_perm('sungjukcall.add_student')  # 이제 정확한 결과
```

---

## 3. request.user — 인증된 사용자 정보 접근

모든 View 함수에서 `request.user`로 현재 요청을 보낸 사용자 객체에 접근할 수 있다. Django 미들웨어(`AuthenticationMiddleware`)가 각 요청마다 자동으로 세션에서 사용자를 찾아 `request.user`에 설정해주기 때문이다.

로그인 상태면 `User` 모델 인스턴스가, 비로그인 상태면 `AnonymousUser` 인스턴스가 들어온다.

```python
def some_view(request):
    # ── 로그인 여부 확인 ────────────────────────────
    if request.user.is_authenticated:
        # 로그인 상태: User 인스턴스
        user = request.user

        print(user.id)            # 1
        print(user.username)      # 'hong'
        print(user.email)         # 'hong@example.com'
        print(user.first_name)    # '길동'
        print(user.last_name)     # '홍'
        print(user.is_staff)      # False (일반 사용자)
        print(user.is_superuser)  # False
        print(user.last_login)    # datetime 객체
        print(user.date_joined)   # datetime 객체

        # 소속 그룹과 권한
        print(user.groups.all())            # QuerySet of Groups
        print(user.user_permissions.all())  # QuerySet of Permissions
        print(user.get_all_permissions())   # set of '앱.권한' 문자열

        # 권한 확인
        print(user.has_perm('polls.add_choice'))   # True or False
        print(user.has_module_perms('polls'))       # 앱 전체 권한 보유 여부

    else:
        # 비로그인 상태: AnonymousUser 인스턴스
        print(request.user)               # AnonymousUser
        print(request.user.is_authenticated)  # False
        print(request.user.is_anonymous)      # True
        print(request.user.username)          # '' (빈 문자열)
        # AnonymousUser는 권한을 전혀 가지지 않음
        print(request.user.has_perm('polls.add_choice'))  # 항상 False
```

템플릿에서도 `request.user`에 접근할 수 있다. `context_processors.py`에 `django.contrib.auth.context_processors.auth`가 등록되어 있어서 별도 설정 없이 모든 템플릿에서 `{{ request.user }}`와 `{{ perms }}`를 사용할 수 있다.

```html
<!-- 어떤 템플릿에서든 사용 가능 -->
{% if request.user.is_authenticated %}
    <p>{{ request.user.username }}님 안녕하세요</p>
    <a href="{% url 'logout' %}">로그아웃</a>
{% else %}
    <a href="{% url 'login' %}">로그인</a>
    <a href="{% url 'register' %}">회원가입</a>
{% endif %}

<!-- 권한 확인 -->
{% if perms.polls.add_choice %}
    <a href="{% url 'choice-create' %}">새 선택지 추가</a>
{% endif %}

{% if request.user.is_superuser %}
    <span class="badge">슈퍼유저</span>
{% elif request.user.is_staff %}
    <span class="badge">스태프</span>
{% endif %}
```

---

## 4. 권한 검사 방법 — FBV 데코레이터와 CBV Mixin

### 4-1. FBV — 데코레이터 방식

슬라이드의 `@login_required`, `@permission_required`가 이것이다. 함수 위에 `@`로 붙이면 View 함수 실행 전에 조건을 확인한다.

```python
from django.contrib.auth.decorators import (
    login_required,
    permission_required,
    user_passes_test
)
from django.shortcuts import render, get_object_or_404

# ── @login_required ────────────────────────────────────
# 로그인하지 않으면 settings.LOGIN_URL 로 리다이렉트
# 기본값: /accounts/login/?next=/현재URL/
@login_required
def student_list(request):
    students = Student.objects.all()
    return render(request, 'students/list.html', {'students': students})

# 로그인 URL을 직접 지정
@login_required(login_url='/my-login/')
def student_list(request):
    ...

# ── @permission_required ───────────────────────────────
# 해당 권한이 없으면 로그인 페이지로 리다이렉트 (기본)
@permission_required('sungjukcall.add_student')
def student_create(request):
    ...

# raise_exception=True: 로그인 페이지 대신 403 Forbidden 반환
# 이미 로그인한 사용자가 권한이 없을 때 적합
@permission_required('sungjukcall.delete_student', raise_exception=True)
def student_delete(request, pk):
    ...

# 여러 권한이 모두 필요한 경우 (AND 조건)
# → 데코레이터를 두 번 쌓으면 됨 (아래에서 위 순서로 실행)
@login_required
@permission_required('sungjukcall.change_student', raise_exception=True)
def student_edit(request, pk):
    ...

# ── @user_passes_test ──────────────────────────────────
# 함수로 직접 조건을 정의할 때 사용
# test 함수가 False를 반환하면 로그인 페이지로 이동
def is_staff_or_superuser(user):
    return user.is_staff or user.is_superuser

@user_passes_test(is_staff_or_superuser)
def admin_dashboard(request):
    ...

# 람다로 간결하게
@user_passes_test(lambda u: u.is_authenticated and u.groups.filter(name='편집자').exists())
def editor_only_view(request):
    ...
```

### 4-2. CBV — Mixin 방식

슬라이드의 `LoginRequiredMixin`, `UserPassesTestMixin`, `PermissionRequiredMixin`이다. 클래스에 상속으로 추가하는 방식이며, FBV 데코레이터와 동일한 기능을 한다.

```python
from django.contrib.auth.mixins import (
    LoginRequiredMixin,
    PermissionRequiredMixin,
    UserPassesTestMixin
)
from django.views.generic import (
    ListView, DetailView, CreateView, UpdateView, DeleteView
)
from django.urls import reverse_lazy

# ── LoginRequiredMixin ─────────────────────────────────
# 로그인 여부만 확인. 비로그인 시 login_url로 이동
class StudentListView(LoginRequiredMixin, ListView):
    model               = Student
    template_name       = 'students/list.html'
    context_object_name = 'students'
    login_url           = '/accounts/login/'   # 기본값. 생략 가능
    redirect_field_name = 'next'               # 로그인 후 돌아올 URL 파라미터명


# ── PermissionRequiredMixin ───────────────────────────
# 특정 권한 보유 여부 확인
class StudentCreateView(PermissionRequiredMixin, CreateView):
    model              = Student
    form_class         = StudentForm
    template_name      = 'students/form.html'
    success_url        = reverse_lazy('student-list')

    # 하나의 권한 문자열
    permission_required = 'sungjukcall.add_student'

    # 여러 권한이 모두 필요한 경우 (AND 조건)
    # permission_required = ['sungjukcall.add_student', 'sungjukcall.change_student']

    # 권한 없을 때 동작 설정
    raise_exception = True   # True: 403 반환 / False: 로그인 페이지로 이동 (기본)

    def handle_no_permission(self):
        """권한 없을 때 커스텀 처리"""
        from django.contrib import messages
        messages.error(self.request, '이 기능을 사용할 권한이 없습니다.')
        return redirect('student-list')


# ── UserPassesTestMixin ───────────────────────────────
# 커스텀 조건으로 접근 제어 (가장 유연한 방법)
# test_func()이 True를 반환해야 View 실행
class StudentUpdateView(UserPassesTestMixin, UpdateView):
    model      = Student
    form_class = StudentForm
    template_name = 'students/form.html'
    success_url   = reverse_lazy('student-list')

    def test_func(self):
        """이 조건이 True여야 접근 가능"""
        student = self.get_object()
        user    = self.request.user

        # 슈퍼유저 또는 스태프이거나, 자신이 만든 데이터인 경우만 허용
        return (
            user.is_superuser or
            user.is_staff or
            student.created_by == user
        )

    def handle_no_permission(self):
        """조건 불충족 시 커스텀 처리"""
        from django.shortcuts import redirect
        from django.contrib import messages
        messages.warning(self.request, '본인이 작성한 데이터만 수정할 수 있습니다.')
        return redirect('student-list')
```

**Mixin 순서 규칙**: 권한/로그인 Mixin은 반드시 가장 왼쪽(첫 번째 상속)에 써야 한다. Python은 상속 목록을 왼쪽에서 오른쪽으로 MRO(Method Resolution Order) 순서로 처리하기 때문에, `LoginRequiredMixin`이 왼쪽에 있어야 `dispatch()` 메서드 오버라이드가 먼저 실행된다.

```python
# 올바른 순서 — LoginRequiredMixin이 가장 먼저
class MyView(LoginRequiredMixin, PermissionRequiredMixin, UpdateView):
    ...

# 잘못된 순서 — UpdateView가 먼저 실행되어 Mixin이 동작하지 않을 수 있음
class MyView(UpdateView, LoginRequiredMixin):
    ...
```

### 4-3. 뷰 안에서 직접 확인

데코레이터나 Mixin 없이 View 코드 안에서 조건을 직접 검사하는 방법이다. 복잡한 로직이 필요하거나 조건에 따라 다른 처리를 해야 할 때 사용한다.

```python
from django.core.exceptions import PermissionDenied
from django.contrib.auth.decorators import login_required
from django.http import HttpResponseForbidden

def student_delete(request, pk):
    # 로그인 여부를 먼저 확인
    if not request.user.is_authenticated:
        from django.shortcuts import redirect
        return redirect(f'/accounts/login/?next={request.path}')

    student = get_object_or_404(Student, pk=pk)

    # 권한 확인 — has_perm() 사용
    if not request.user.has_perm('sungjukcall.delete_student'):
        raise PermissionDenied   # 403 HTTP 에러 발생 → 403.html 렌더링

    # 객체 단위 권한 확인 (이 특정 데이터에 대한 접근 권한)
    if not (request.user.is_superuser or
            getattr(student, 'created_by', None) == request.user):
        raise PermissionDenied

    if request.method == 'POST':
        student.delete()
        return redirect('student-list')

    return render(request, 'students/confirm_delete.html', {'student': student})
```

---

## 5. 기본 인증 — django.contrib.auth

슬라이드 하단의 인증 흐름이다.

```
urls (accounts/)
   ↓
django.contrib.auth
   ├─ Login   → 로그인 로직 내장 (개발자는 템플릿만 제공)
   ├─ Logout  → 로그아웃 로직 내장
   └─ register (기본 제공 없음 → 직접 구현 필요)
   ↓
미리 구현된 App 구현 (비즈니스 로직은 Django가 처리)
   ↓
인터페이스 자유부여 (개발자가 HTML 디자인만 결정)
   ├─ Login.html
   ├─ Logout.html
   └─ Register.html
```

Django가 제공하는 것과 개발자가 만들어야 하는 것을 명확히 구분하면 다음과 같다.

```
Django가 이미 만들어준 것:
  - 로그인 로직 (인증 확인, 세션 생성)
  - 로그아웃 로직 (세션 제거)
  - 비밀번호 변경, 재설정 이메일 발송

개발자가 만들어야 하는 것:
  - HTML 템플릿 (login.html, 비밀번호 변경 페이지 등)
  - 회원가입 (register) — 전혀 제공하지 않음
  - 커스텀 필드가 필요한 프로필 관리
```

### urls.py 설정

```python
# 프로젝트 루트 urls.py
from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path('admin/', admin.site.urls),

    # django.contrib.auth.urls 포함: 로그인/로그아웃/비밀번호 관련 URL 자동 등록
    path('accounts/', include('django.contrib.auth.urls')),

    # 커스텀 앱 URL (회원가입 등 직접 만든 것)
    path('accounts/', include('accounts.urls')),

    path('', include('students.urls')),
]
```

`django.contrib.auth.urls`를 포함하면 자동으로 등록되는 URL 목록:

|URL 패턴|View 이름|설명|
|---|---|---|
|`accounts/login/`|`login`|로그인 페이지|
|`accounts/logout/`|`logout`|로그아웃 처리|
|`accounts/password_change/`|`password_change`|비밀번호 변경|
|`accounts/password_change/done/`|`password_change_done`|변경 완료 안내|
|`accounts/password_reset/`|`password_reset`|비밀번호 재설정 요청|
|`accounts/password_reset/done/`|`password_reset_done`|이메일 발송 안내|
|`accounts/reset/<uidb64>/<token>/`|`password_reset_confirm`|재설정 폼|
|`accounts/reset/done/`|`password_reset_complete`|재설정 완료|

---

## 6. 로그인 구현

### 방법 1 — Django 내장 View 사용 (권장)

Django가 모든 로직을 처리한다. 개발자는 템플릿만 만들면 된다.

```python
# accounts/urls.py
from django.contrib.auth import views as auth_views
from django.urls import path

urlpatterns = [
    path('login/',
         auth_views.LoginView.as_view(template_name='accounts/login.html'),
         name='login'),
    path('logout/',
         auth_views.LogoutView.as_view(),
         name='logout'),
]
```

```python
# settings.py — 로그인 관련 URL 설정
LOGIN_URL          = '/accounts/login/'   # 로그인 페이지 URL (기본값)
LOGIN_REDIRECT_URL = '/'                  # 로그인 성공 후 이동할 URL
LOGOUT_REDIRECT_URL = '/accounts/login/'  # 로그아웃 후 이동할 URL
```

```html
<!-- templates/accounts/login.html -->
<!DOCTYPE html>
<html lang="ko">
<head>
    <title>로그인</title>
    <link rel="stylesheet"
          href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css">
</head>
<body>
<div class="container mt-5" style="max-width: 420px;">
    <h2 class="mb-4 fw-bold">로그인</h2>

    <!-- 로그인 실패 메시지 표시 -->
    {% if form.errors %}
    <div class="alert alert-danger">
        아이디 또는 비밀번호가 올바르지 않습니다. 다시 확인해주세요.
    </div>
    {% endif %}

    <!-- 권한 없어서 로그인 페이지로 왔을 때 메시지 -->
    {% if next %}
    <div class="alert alert-warning">
        이 페이지에 접근하려면 로그인이 필요합니다.
    </div>
    {% endif %}

    <form method="post">
        {% csrf_token %}
        <div class="mb-3">
            <label for="id_username" class="form-label">아이디</label>
            <input type="text" name="username" id="id_username"
                   class="form-control" autofocus required>
        </div>
        <div class="mb-3">
            <label for="id_password" class="form-label">비밀번호</label>
            <input type="password" name="password" id="id_password"
                   class="form-control" required>
        </div>

        <!-- next: 로그인 후 원래 페이지로 돌아가기 위한 히든 필드 -->
        {% if next %}
            <input type="hidden" name="next" value="{{ next }}">
        {% endif %}

        <div class="d-grid">
            <button type="submit" class="btn btn-primary">로그인</button>
        </div>
    </form>

    <hr>
    <div class="text-center">
        <a href="{% url 'register' %}">회원가입</a> &nbsp;|&nbsp;
        <a href="{% url 'password_reset' %}">비밀번호 찾기</a>
    </div>
</div>
</body>
</html>
```

### 방법 2 — 직접 구현 (FBV)

로그인 로직을 직접 제어해야 하거나 추가 처리(예: 로그인 로그 기록, 소셜 로그인 연동)가 필요한 경우에 사용한다.

```python
# accounts/views.py
from django.contrib.auth import authenticate, login, logout
from django.contrib.auth.forms import AuthenticationForm
from django.shortcuts import render, redirect

def login_view(request):
    # 이미 로그인된 사용자가 로그인 페이지로 오면 홈으로 이동
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        # AuthenticationForm: Django 내장 로그인 폼
        # request를 첫 번째 인자로 받아야 함 (일반 폼과 다름)
        form = AuthenticationForm(request, data=request.POST)

        if form.is_valid():
            username = form.cleaned_data.get('username')
            password = form.cleaned_data.get('password')

            # authenticate: DB에서 사용자 확인 + 비밀번호 검증
            # 성공하면 User 객체 반환, 실패하면 None 반환
            user = authenticate(request, username=username, password=password)

            if user is not None:
                if user.is_active:
                    # login: 세션에 사용자 정보 저장
                    # → sessionid 쿠키가 브라우저에 전송됨
                    login(request, user)

                    # next 파라미터가 있으면 해당 URL로, 없으면 홈으로
                    next_url = request.GET.get('next') or request.POST.get('next') or 'home'
                    return redirect(next_url)
                else:
                    # 계정이 비활성화된 경우 (is_active=False)
                    form.add_error(None, '비활성화된 계정입니다. 관리자에게 문의하세요.')
    else:
        form = AuthenticationForm()

    return render(request, 'accounts/login.html', {'form': form})


def logout_view(request):
    # POST로만 로그아웃 처리 (CSRF 보호 + GET 요청으로 로그아웃 방지)
    if request.method == 'POST':
        logout(request)  # 세션에서 사용자 정보 삭제 + 세션 ID 무효화
        return redirect('login')
    return redirect('home')
```

---

## 7. 회원가입 (Register) 구현

`register`는 Django가 기본으로 제공하지 않는다. `UserCreationForm`을 확장해서 직접 만든다. `UserCreationForm`은 `username`, `password1`, `password2` 세 필드를 기본 제공하고 비밀번호 일치 여부와 강도 검증을 자동으로 처리한다.

```python
# accounts/forms.py
from django import forms
from django.contrib.auth.models import User
from django.contrib.auth.forms import UserCreationForm

class RegisterForm(UserCreationForm):
    """Django 내장 UserCreationForm 확장"""
    email      = forms.EmailField(
        required=True,
        label='이메일',
        help_text='유효한 이메일 주소를 입력하세요.'
    )
    first_name = forms.CharField(
        max_length=50,
        required=False,
        label='이름',
        widget=forms.TextInput(attrs={'placeholder': '이름 (선택)'})
    )

    class Meta:
        model  = User
        fields = ('username', 'email', 'first_name', 'password1', 'password2')
        labels = {'username': '아이디'}
        help_texts = {'username': None}  # 기본 도움말 문구 제거

    def clean_email(self):
        """이메일 중복 검사 커스텀 검증"""
        email = self.cleaned_data.get('email')
        if User.objects.filter(email=email).exists():
            raise forms.ValidationError('이미 사용 중인 이메일 주소입니다.')
        return email

    def clean_username(self):
        """아이디에 특수문자 제한 등 커스텀 검증"""
        username = self.cleaned_data.get('username')
        if len(username) < 4:
            raise forms.ValidationError('아이디는 4자 이상이어야 합니다.')
        return username
```

```python
# accounts/views.py
from django.contrib.auth import login
from .forms import RegisterForm

def register_view(request):
    if request.user.is_authenticated:
        return redirect('home')

    if request.method == 'POST':
        form = RegisterForm(request.POST)
        if form.is_valid():
            # save(): User 객체 생성 + 비밀번호 자동 해시화
            # UserCreationForm.save()가 내부적으로 set_password() 호출
            user = form.save()

            # 회원가입 직후 자동 로그인 처리
            login(request, user)

            return redirect('home')
    else:
        form = RegisterForm()

    return render(request, 'accounts/register.html', {'form': form})
```

```html
<!-- templates/accounts/register.html -->
<form method="post">
    {% csrf_token %}

    {% for field in form %}
    <div class="mb-3">
        <label for="{{ field.id_for_label }}" class="form-label">
            {{ field.label }}
            {% if field.field.required %}<span class="text-danger">*</span>{% endif %}
        </label>
        {{ field }}
        {% if field.help_text %}
            <small class="text-muted">{{ field.help_text }}</small>
        {% endif %}
        {% if field.errors %}
            {% for error in field.errors %}
                <div class="text-danger small">{{ error }}</div>
            {% endfor %}
        {% endif %}
    </div>
    {% endfor %}

    <button type="submit" class="btn btn-success w-100">회원가입</button>
</form>
<p class="text-center mt-3">
    이미 계정이 있으신가요? <a href="{% url 'login' %}">로그인</a>
</p>
```

```python
# accounts/urls.py
from django.urls import path
from . import views

urlpatterns = [
    path('login/',    views.login_view,    name='login'),
    path('logout/',   views.logout_view,   name='logout'),
    path('register/', views.register_view, name='register'),
]
```

---

## 8. 인증 흐름 상세 — 세션과 쿠키

로그인이 내부적으로 어떻게 동작하는지 단계별로 이해하면 디버깅과 보안 설계가 쉬워진다.

```
로그인 과정:

①  사용자가 username='hong', password='1234' 입력 후 POST 전송

②  authenticate(request, username='hong', password='1234')
    → DB에서 username='hong' 인 User 조회
    → 저장된 해시: pbkdf2_sha256$390000$salt$hash...
    → 입력된 '1234'를 같은 방식으로 해시
    → 두 해시 비교 → 일치하면 User 객체, 불일치하면 None 반환

③  login(request, user)
    → 새 세션 생성 (서버 측 DB 또는 캐시에 저장)
    → 세션 ID (예: 'abc123xyz') 생성
    → Set-Cookie: sessionid=abc123xyz; HttpOnly 헤더로 브라우저에 전송
    → 세션에 user_id 저장: session['_auth_user_id'] = str(user.pk)

④  이후 모든 요청
    → 브라우저가 자동으로 Cookie: sessionid=abc123xyz 전송
    → Django AuthenticationMiddleware가 세션 ID로 사용자 조회
    → request.user = User.objects.get(pk=세션의 user_id)

로그아웃 과정:

①  logout(request) 호출
    → 서버의 세션 데이터 삭제
    → 브라우저의 sessionid 쿠키 만료 처리

②  다음 요청
    → 쿠키가 없거나 세션이 없음
    → request.user = AnonymousUser()
```

세션 관련 보안 설정:

```python
# settings.py
SESSION_COOKIE_AGE     = 1209600   # 세션 유지 시간 (초), 기본 2주
SESSION_EXPIRE_AT_BROWSER_CLOSE = False  # True면 브라우저 닫으면 세션 만료
SESSION_COOKIE_SECURE  = True      # HTTPS에서만 쿠키 전송 (운영 환경 필수)
SESSION_COOKIE_HTTPONLY = True     # JavaScript에서 쿠키 접근 불가 (XSS 방어)
```

---

## 9. 커스텀 User 모델

Django 기본 User 모델(`username`, `password`, `email` 등)로 부족할 때 커스텀 User 모델을 만든다. 전화번호, 프로필 이미지, 생년월일 등 추가 필드가 필요한 경우다.

**반드시 프로젝트 시작 시점에 설정해야 한다.** 마이그레이션이 진행된 후에 `AUTH_USER_MODEL`을 변경하면 기존 테이블과 관계가 꼬여 복구하기 매우 어렵다.

```python
# accounts/models.py
from django.contrib.auth.models import AbstractUser
from django.db import models

class CustomUser(AbstractUser):
    """
    AbstractUser를 상속하면 기본 User의 모든 필드와 메서드를 그대로 사용하면서
    추가 필드만 정의할 수 있다.
    AbstractBaseUser: 더 낮은 레벨에서 완전히 커스텀할 때 사용 (고급)
    """
    phone = models.CharField(
        max_length=20, blank=True,
        verbose_name='전화번호',
        help_text='010-0000-0000 형식'
    )
    birth_date = models.DateField(
        null=True, blank=True,
        verbose_name='생년월일'
    )
    profile_image = models.ImageField(
        upload_to='profiles/%Y/%m/',
        blank=True, null=True,
        verbose_name='프로필 이미지'
    )
    bio = models.TextField(
        blank=True,
        verbose_name='자기소개'
    )

    class Meta:
        verbose_name        = '사용자'
        verbose_name_plural = '사용자 목록'

    def __str__(self):
        return f'{self.username} ({self.get_full_name() or self.email})'

    def get_full_name(self):
        return f'{self.last_name}{self.first_name}'.strip()
```

```python
# settings.py — 반드시 첫 migrate 전에 설정
AUTH_USER_MODEL = 'accounts.CustomUser'
```

```python
# 다른 앱에서 User 모델을 참조할 때 하드코딩 금지
# 이유: 나중에 커스텀 모델로 바꾸면 모든 곳을 수정해야 함

# 잘못된 방법 (하드코딩)
from django.contrib.auth.models import User

# 올바른 방법: get_user_model() 사용
from django.contrib.auth import get_user_model
User = get_user_model()   # settings.AUTH_USER_MODEL 을 참조함

# models.py에서 ForeignKey로 참조할 때
from django.conf import settings

class Post(models.Model):
    author = models.ForeignKey(
        settings.AUTH_USER_MODEL,  # 'accounts.CustomUser'
        on_delete=models.CASCADE
    )
```

```python
# accounts/admin.py — 커스텀 User 관리자 등록
from django.contrib import admin
from django.contrib.auth.admin import UserAdmin
from .models import CustomUser

@admin.register(CustomUser)
class CustomUserAdmin(UserAdmin):
    # 기본 UserAdmin에 커스텀 필드 추가
    fieldsets = UserAdmin.fieldsets + (
        ('추가 정보', {'fields': ('phone', 'birth_date', 'profile_image', 'bio')}),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        ('추가 정보', {'fields': ('phone', 'birth_date')}),
    )
    list_display = ['username', 'email', 'phone', 'is_staff', 'is_active']
    search_fields = ['username', 'email', 'phone']
```

---

## 10. 전체 흐름 정리

### 인증 흐름

```
브라우저 → POST /accounts/login/ (username, password)
                 ↓
          AuthenticationForm.is_valid()
                 ↓
          authenticate(username, password)
          → DB 조회 → 비밀번호 해시 비교
                 ↓
          User 반환 (성공) / None (실패)
                 ↓
          login(request, user)
          → 세션 생성 → sessionid 쿠키 → 브라우저
                 ↓
          redirect(next or LOGIN_REDIRECT_URL)
```

### 권한 확인 흐름

```
브라우저 → GET /students/new/
                 ↓
          @login_required  또는  LoginRequiredMixin
          ├─ 비로그인: redirect → /accounts/login/?next=/students/new/
          └─ 로그인: 다음 단계
                 ↓
          @permission_required  또는  PermissionRequiredMixin
          ├─ 권한 없음:
          │    raise_exception=True  → 403 Forbidden
          │    raise_exception=False → redirect → /accounts/login/
          └─ 권한 있음: View 실행
                 ↓
          View 실행 → Model 쿼리 → Template 렌더링
                 ↓
          HTTP Response → 브라우저
```

### 핵심 요약표

|항목|FBV 방식|CBV 방식|
|---|---|---|
|로그인 필수|`@login_required`|`LoginRequiredMixin`|
|특정 권한 필수|`@permission_required('앱.권한')`|`PermissionRequiredMixin`|
|커스텀 조건|`@user_passes_test(함수)`|`UserPassesTestMixin` + `test_func()`|
|현재 사용자|`request.user`|`self.request.user`|
|로그인 여부|`request.user.is_authenticated`|`self.request.user.is_authenticated`|
|권한 보유 여부|`request.user.has_perm('앱.권한')`|`self.request.user.has_perm(...)`|
|템플릿 권한 확인|`{% if perms.앱.권한 %}`|동일|
|로그인 처리|`login(request, user)`|동일|
|로그아웃 처리|`logout(request)`|동일|
|사용자 인증|`authenticate(request, username, password)`|동일|