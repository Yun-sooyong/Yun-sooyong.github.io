---
title: 055 React
tag:
  - 헬스케어 ai
  - react
description: 260721 수업 내용 정리
---

# React.js 

React는 Facebook(현 Meta)이 만든 JavaScript 라이브러리다. 웹 페이지의 UI를 컴포넌트(Component) 단위로 쪼개서 만들고, 데이터가 바뀔 때마다 필요한 부분만 자동으로 업데이트한다. 기존 HTML 템플릿 방식을 React가 대체하면서 프론트엔드 개발의 표준이 되었다.

---

## 1. 개발 환경 설정 — WSL, Node.js, npm

### WSL (Windows Subsystem for Linux)

Windows에서 Linux 환경을 그대로 사용할 수 있게 해주는 기능이다. React 개발은 macOS나 Linux 환경에서 더 원활하게 동작하는 도구들이 많기 때문에, Windows 사용자는 WSL을 통해 Linux 환경을 구축하는 것이 권장된다.

```bash
# WSL + Ubuntu 22.04 LTS 설치 (PowerShell 관리자 모드)
wsl --install -d Ubuntu-22.04

# 설치 확인
wsl --list --verbose

# Ubuntu 터미널 진입
wsl
```

### Node.js와 npm

React 앱을 실행하고 빌드하려면 Node.js가 반드시 필요하다. Node.js는 JavaScript를 브라우저 밖에서 실행할 수 있게 해주는 런타임이고, npm(Node Package Manager)은 그 안에 포함된 패키지 관리 도구다.

```
Node.js 설치하면 함께 설치되는 것:
  npm        — 패키지 설치/관리 명령어
  npx        — 패키지를 설치 없이 바로 실행
  node_modules/ — 설치된 패키지들이 저장되는 폴더
```

```bash
# Node.js 설치 (nvm 사용 권장 — 버전 관리 편리)
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.7/install.sh | bash
source ~/.bashrc
nvm install --lts        # 최신 LTS 버전 설치
nvm use --lts

# 버전 확인
node -v    # v20.x.x
npm -v     # 10.x.x
```

### package.json

npm으로 만든 프로젝트의 **설명서이자 설정 파일**이다. 프로젝트 이름, 버전, 사용하는 패키지 목록, 실행 스크립트 등이 담긴다.

```json
{
  "name": "my-react-app",
  "version": "0.1.0",
  "dependencies": {
    "react": "^18.2.0",
    "react-dom": "^18.2.0"
  },
  "devDependencies": {
    "vite": "^5.0.0"
  },
  "scripts": {
    "dev":   "vite",         // 개발 서버 실행
    "build": "vite build",   // 프로덕션 빌드
    "preview": "vite preview"
  }
}
```

`npm install`을 실행하면 `package.json`에 명시된 패키지들을 모두 `node_modules/`에 설치한다. `package-lock.json`은 설치된 정확한 버전을 고정해서 팀원들이 같은 환경을 유지하게 한다.

---

## 2. 빌드 도구 — Webpack, Vite, Babel

React 코드는 브라우저가 바로 이해하지 못하는 형식(JSX, ES6+)으로 작성된다. 빌드 도구는 이 코드를 브라우저가 이해할 수 있는 형태로 변환하고 최적화하는 역할을 한다.

```
React 개발 흐름:

개발자 코드 (.jsx, .tsx, ES6+)
        ↓ 빌드 도구
브라우저가 이해하는 코드 (.js, .css, .html)
```

### Babel

JavaScript 최신 문법(ES6+)과 JSX를 구형 브라우저에서도 동작하는 코드로 변환(트랜스파일)한다. React + Babel 조합에서 Babel이 JSX 문법을 `React.createElement()` 호출로 바꿔준다.

```jsx
// Babel 변환 전 (개발자가 작성)
const element = <h1>Hello</h1>;

// Babel 변환 후 (브라우저가 실행)
const element = React.createElement('h1', null, 'Hello');
```

### Webpack

모듈 번들러다. 수십~수백 개의 `.js`, `.css`, `.png` 파일들을 분석해서 의존성 그래프를 만들고, 최적화된 번들 파일 몇 개로 묶어준다. Create React App(CRA)의 기본 빌드 도구였으나 설정이 복잡하고 느린 편이다.

### Vite (현재 권장)

2021년 이후 React 생태계에서 Webpack을 빠르게 대체하고 있는 빌드 도구다. 개발 서버 시작이 Webpack보다 10~100배 빠르다. 그 이유는 개발 중에는 번들링 없이 브라우저의 ES Module을 그대로 사용하고, 변경된 파일만 즉시 교체(HMR, Hot Module Replacement)하기 때문이다.

```bash
# Vite로 React 프로젝트 생성
npm create vite@latest my-app -- --template react
cd my-app
npm install
npm run dev    # 개발 서버 시작 → http://localhost:5173
```

### esbuild와 Rollup

esbuild는 Go로 작성된 극도로 빠른 번들러로, Vite가 내부적으로 사용한다. Rollup은 라이브러리 배포에 적합한 번들러로, 역시 Vite의 프로덕션 빌드에 사용된다.

---

## 3. Frontend vs Backend

웹 서비스는 크게 두 영역으로 나뉜다.

```
[Customer]     [Web Server]    [App Server]    [Database]
브라우저  ←→  웹 서버      ←→  앱 서버      ←→  데이터베이스

Frontend: Customer ↔ Web Server 사이 (브라우저에서 실행되는 모든 것)
Backend:  Web Server ↔ Database 사이 (서버에서 실행되는 모든 것)
```

|구분|Frontend|Backend|
|---|---|---|
|실행 환경|사용자의 브라우저|서버|
|언어|HTML, CSS, JavaScript|Python, Java, Node.js 등|
|역할|화면 표시, 사용자 상호작용|데이터 처리, DB 연동, 비즈니스 로직|
|React의 위치|Frontend|—|

React는 순수 Frontend 라이브러리다. 서버와는 REST API 또는 GraphQL로 데이터를 주고받는다.

---

## 4. SPA (Single Page Application)

전통적인 웹은 페이지를 이동할 때마다 서버에서 새 HTML을 받아와 전체 페이지를 다시 그렸다. SPA는 처음에 HTML, CSS, JavaScript를 한 번만 받아오고, 이후 페이지 이동 시 **필요한 데이터만 서버에서 받아와 JavaScript로 화면을 바꾼다**.

```
전통적인 MPA (Multi Page Application):
  페이지 이동 → 서버 요청 → 전체 HTML 다시 받음 → 깜빡임, 느림

SPA (Single Page Application):
  페이지 이동 → JS가 화면 일부만 교체 → 깜빡임 없음, 빠름
              → 필요한 데이터만 API로 요청

React = SPA를 만드는 대표적인 도구
```

**SPA의 단점**: 초기 로딩 시 JavaScript를 전부 받아야 해서 첫 화면이 느릴 수 있다. 검색 엔진 최적화(SEO)에도 불리하다. 이를 해결하기 위해 Next.js(SSR, SSG)를 사용하기도 한다.

---

## 5. HTML vs JSX — JS + HTML = JSX

React에서는 JavaScript 파일 안에 HTML처럼 생긴 코드를 직접 쓸 수 있다. 이것이 **JSX(JavaScript XML)** 다.

```
JavaScript + HTML ≡ JSX (ReactJS)
```

JSX는 HTML이 아니다. JavaScript 코드이며, Babel이 이것을 `React.createElement()` 호출로 변환한다.

```jsx
// JSX 문법
function Greeting() {
  const name = '수용';
  return (
    <div className="greeting">   {/* class → className */}
      <h1>안녕하세요, {name}님!</h1>   {/* {} 안에 JS 표현식 */}
      <p>React를 배워봅시다.</p>
    </div>
  );
}
```

### HTML vs JSX 주요 차이점

|HTML|JSX|이유|
|---|---|---|
|`class="..."`|`className="..."`|`class`는 JS 예약어|
|`for="..."`|`htmlFor="..."`|`for`는 JS 예약어|
|`<br>`|`<br />`|반드시 자기 닫기|
|`onclick="..."`|`onClick={...}`|camelCase 이벤트명|
|주석: `<!-- -->`|주석: `{/* */}`|JS 문법 사용|

---

## 6. HTML 프로그래밍 방식과 React의 차이

### HTML 프로그래밍의 세 가지 방식

전통적인 웹 개발에서 HTML을 다루는 방식은 세 가지로 구분된다.

**Tag Based Programming**: HTML 태그를 직접 작성해서 정적인 구조를 만드는 방식이다. 내용이 변경되지 않는 정적 페이지에 적합하다.

**GUI Programming**: 버튼, 입력창, 드롭다운 같은 UI 요소들을 배치하고 구성하는 방식이다. 사용자와 상호작용하는 인터페이스를 만든다.

**Event Based Programming**: 사용자 행동(클릭, 키 입력, 마우스 이동)에 반응해서 동작하는 방식이다. `addEventListener`로 이벤트를 등록하고 콜백 함수를 실행한다.

### React의 방식 — Component Base Programming

React는 위 세 가지를 **컴포넌트(Component)** 라는 단위로 통합한다.

```
HTML 방식:
  구조(HTML) + 스타일(CSS) + 동작(JS)을 각각 분리

React 방식:
  하나의 컴포넌트 안에 구조 + 스타일 + 동작을 모두 포함
  → 재사용, 유지보수, 테스트가 쉬워짐
```

---

## 7. Virtual DOM — 효율적인 화면 업데이트

### Real DOM의 문제

브라우저의 Real DOM은 조작할 때마다 **레이아웃 재계산, 리페인트** 과정이 발생해서 느리다. 화면의 작은 부분 하나를 바꿔도 전체 DOM 트리가 영향을 받을 수 있다.

### Virtual DOM의 해결책

React는 Real DOM의 **가벼운 복사본(Virtual DOM)** 을 메모리에 유지한다. 데이터가 바뀌면 먼저 Virtual DOM을 업데이트하고, 이전 Virtual DOM과 새 Virtual DOM을 **비교(Diffing)** 해서 실제로 바뀐 부분만 Real DOM에 적용한다.

```
데이터 변경 발생
        ↓
새 Virtual DOM 생성
        ↓
이전 Virtual DOM과 비교 (Reconciliation / Diffing)
        ↓
변경된 부분만 Real DOM에 반영 (Patch)
        ↓
브라우저가 화면 업데이트
```

```
Virtual DOM 업데이트: useRef()로 참조
Real DOM 업데이트:   Event, setState, useEffect, useContext, useReducer
```

이 과정을 **Reconciliation(재조정)** 이라고 한다. React 18에서는 이 과정을 더 효율적으로 만든 **Concurrent Mode**가 도입되었다.

---

## 8. 컴포넌트 (Component)

컴포넌트는 React에서 UI를 구성하는 **독립적이고 재사용 가능한 단위**다. 마치 레고 블록처럼, 작은 컴포넌트들을 조합해서 복잡한 UI를 만든다.

```
App (루트 컴포넌트)
  ├─ Header
  │    └─ NavBar
  ├─ Main
  │    ├─ Sidebar
  │    └─ Content
  │         ├─ Card
  │         └─ Card
  └─ Footer
```

### 컴포넌트의 구성 요소

컴포넌트는 크게 네 가지 개념으로 구성된다.

**Props**: 부모 컴포넌트에서 자식 컴포넌트로 전달하는 데이터다. 읽기 전용이며, 컴포넌트 외부에서 주어지는 값이다.

**State**: 컴포넌트 내부에서 관리하는 데이터다. state가 변경되면 해당 컴포넌트와 자식 컴포넌트가 자동으로 다시 렌더링된다.

**JSX**: 컴포넌트가 화면에 출력할 내용을 표현하는 문법이다.

**Side Effect**: 네트워크 요청, 타이머, DOM 직접 조작처럼 React 렌더링 흐름 밖에서 일어나는 작업이다. `useEffect`로 처리한다.

```jsx
// Props와 State를 함께 사용하는 예시
function Counter({ initialCount }) {   // initialCount: Props (부모에서 전달)
  const [count, setCount] = useState(initialCount);  // count: State (내부 관리)

  return (
    <div>
      <p>현재 카운트: {count}</p>
      <button onClick={() => setCount(count + 1)}>+1</button>
    </div>
  );
}

// 사용
<Counter initialCount={0} />   // Props로 초기값 전달
```

### Component Tree 구조와 Props & State 흐름

컴포넌트들은 **트리(Tree)** 구조를 이룬다. Props는 부모에서 자식으로만 흐른다(단방향). State는 해당 컴포넌트 내부에서 관리되지만, 여러 컴포넌트가 같은 데이터를 공유해야 할 때는 공통 조상으로 state를 올려두는 **State 끌어올리기(Lifting State Up)** 패턴을 사용한다.

```
Props 흐름 (단방향 ↓):
  App [state: user]
    ↓ props: user
  Header [props.user]
    ↓ props: user.name
  NavBar [props.user.name]

State 변경 → 해당 컴포넌트 + 모든 자식 리렌더링
```

---

## 9. 컴포넌트 구현 방법 — Class vs Function

### Class 컴포넌트 (구형)

ES6 클래스 문법으로 작성하는 방식이다. `render()` 메서드가 필수이며, state와 생명주기 메서드를 내장하고 있다. 현재는 잘 사용하지 않지만 레거시 코드 이해를 위해 알아두어야 한다.

```jsx
import React, { Component } from 'react';

class Counter extends Component {
  constructor(props) {
    super(props);
    this.state = { count: 0 };   // state 초기화
  }

  componentDidMount() {          // 마운트 후 실행 (생명주기)
    console.log('컴포넌트가 화면에 나타났다');
  }

  render() {                     // 필수 메서드 — JSX 반환
    return (
      <div>
        <p>{this.state.count}</p>
        <button onClick={() => this.setState({ count: this.state.count + 1 })}>
          +1
        </button>
      </div>
    );
  }
}
```

### Function 컴포넌트 + Hooks (현재 표준)

함수 형태로 작성하는 방식이다. React 16.8에서 **Hooks**가 도입되면서 함수 컴포넌트에서도 state와 생명주기를 사용할 수 있게 되었고, 현재는 Function 컴포넌트가 표준이다.

```jsx
import { useState, useEffect } from 'react';

function Counter({ initialCount = 0 }) {
  const [count, setCount] = useState(initialCount);   // useState Hook

  useEffect(() => {                                    // 생명주기 Hook
    console.log('count가 바뀌었다:', count);
    document.title = `카운트: ${count}`;
  }, [count]);   // count가 바뀔 때마다 실행

  return (
    <div>
      <p>{count}</p>
      <button onClick={() => setCount(count + 1)}>+1</button>
    </div>
  );
}
```

### Class vs Function 컴포넌트 비교

|구분|Class 컴포넌트|Function 컴포넌트|
|---|---|---|
|문법|`class Foo extends Component`|`function Foo()` 또는 화살표 함수|
|State|`this.state`, `this.setState()`|`useState()` Hook|
|생명주기|`componentDidMount` 등 별도 메서드|`useEffect()` Hook으로 통합|
|코드 양|많음|적음|
|현재 권장|❌ (레거시)|✅ (표준)|

---

## 10. 주요 Hooks

Hooks는 함수 컴포넌트에서 React의 기능을 사용할 수 있게 해주는 함수들이다. 모두 `use`로 시작한다.

### useState — 상태 관리

```jsx
const [state, setState] = useState(초기값);

// 예시
const [count, setCount]     = useState(0);
const [name,  setName]      = useState('');
const [isOpen, setIsOpen]   = useState(false);
const [items,  setItems]    = useState([]);

// 객체 state 업데이트 (스프레드 연산자 필수)
const [user, setUser] = useState({ name: '홍길동', age: 30 });
setUser(prev => ({ ...prev, age: 31 }));   // age만 변경
```

### useEffect — 사이드 이펙트 처리

렌더링 후 실행되는 코드를 정의한다. API 호출, 이벤트 리스너 등록, 타이머 설정 등에 사용한다.

```jsx
useEffect(() => {
  // 실행할 코드
  fetch('https://api.example.com/data')
    .then(res => res.json())
    .then(data => setData(data));

  // 정리 함수 (컴포넌트 제거 시 실행)
  return () => {
    console.log('정리 실행');
  };
}, [의존성]);  // 의존성 배열

// 의존성 배열 세 가지 패턴:
useEffect(() => {...});           // 매 렌더링 후 실행
useEffect(() => {...}, []);       // 마운트 시 한 번만 실행
useEffect(() => {...}, [count]);  // count가 바뀔 때마다 실행
```

### useRef — DOM 참조 / 렌더링 없는 값 저장

Virtual DOM 업데이트를 거치지 않고 Real DOM에 직접 접근하거나, 렌더링을 유발하지 않는 값을 저장할 때 사용한다.

```jsx
const inputRef = useRef(null);

// DOM 직접 접근
<input ref={inputRef} />
<button onClick={() => inputRef.current.focus()}>포커스</button>

// 렌더링 없는 값 저장 (타이머 ID, 이전 값 등)
const timerRef = useRef(null);
timerRef.current = setTimeout(() => {...}, 1000);
```

### useContext — 전역 상태 공유

Props를 여러 단계로 전달하지 않고(Prop Drilling 방지), 멀리 떨어진 컴포넌트에 직접 데이터를 전달한다.

```jsx
// 1. Context 생성
const ThemeContext = React.createContext('light');

// 2. Provider로 값 공급
function App() {
  return (
    <ThemeContext.Provider value="dark">
      <DeepChild />   {/* 몇 단계 아래에 있어도 접근 가능 */}
    </ThemeContext.Provider>
  );
}

// 3. 어느 자식에서든 사용
function DeepChild() {
  const theme = useContext(ThemeContext);   // 'dark'
  return <div className={theme}>...</div>;
}
```

### useReducer — 복잡한 상태 관리

state 업데이트 로직이 복잡할 때 useState 대신 사용한다. Redux와 유사한 패턴으로 상태를 관리한다.

```jsx
function reducer(state, action) {
  switch (action.type) {
    case 'INCREMENT': return { count: state.count + 1 };
    case 'DECREMENT': return { count: state.count - 1 };
    case 'RESET':     return { count: 0 };
    default:          return state;
  }
}

function Counter() {
  const [state, dispatch] = useReducer(reducer, { count: 0 });

  return (
    <div>
      <p>{state.count}</p>
      <button onClick={() => dispatch({ type: 'INCREMENT' })}>+</button>
      <button onClick={() => dispatch({ type: 'DECREMENT' })}>-</button>
      <button onClick={() => dispatch({ type: 'RESET' })}>초기화</button>
    </div>
  );
}
```

### useContext + useReducer 결합 사용

복잡한 전역 상태 관리가 필요할 때 두 Hook을 결합한다. Redux 없이 유사한 기능을 구현할 수 있다.

```jsx
// Context + Reducer 결합 패턴
const CounterContext = React.createContext(null);

function CounterProvider({ children }) {
  const [state, dispatch] = useReducer(reducer, { count: 0 });

  return (
    <CounterContext.Provider value={{ state, dispatch }}>
      {children}
    </CounterContext.Provider>
  );
}

// 어디서든 사용
function AnyChild() {
  const { state, dispatch } = useContext(CounterContext);
  return <button onClick={() => dispatch({ type: 'INCREMENT' })}>
    {state.count}
  </button>;
}
```

---

## 11. 렌더링 (Render)

**렌더링**은 컴포넌트 함수가 호출되어 JSX를 반환하고 이것이 화면에 그려지는 과정이다.

렌더링이 발생하는 조건:

- `setState`로 state가 변경될 때
- 부모로부터 받는 Props가 변경될 때
- 부모 컴포넌트가 리렌더링될 때
- `useContext`로 구독한 Context 값이 변경될 때

```
렌더링 흐름:

1. 트리거: state 변경, props 변경 등
2. 렌더 단계: React가 컴포넌트 함수를 호출 → JSX 반환 → Virtual DOM 생성
3. 커밋 단계: Virtual DOM과 Real DOM 비교 → 변경된 부분만 Real DOM 업데이트
4. 브라우저가 화면에 반영
```

불필요한 리렌더링을 방지하는 방법으로 `React.memo`, `useMemo`, `useCallback`이 있다.

---

## 12. 첫 번째 React 앱 만들기

```bash
# 프로젝트 생성
npm create vite@latest my-first-app -- --template react
cd my-first-app
npm install
npm run dev
```

```
프로젝트 구조:
my-first-app/
  ├─ public/
  │    └─ vite.svg
  ├─ src/
  │    ├─ App.jsx        ← 루트 컴포넌트
  │    ├─ App.css
  │    ├─ main.jsx       ← 진입점 (ReactDOM.render)
  │    └─ index.css
  ├─ index.html          ← SPA의 단일 HTML 파일
  ├─ package.json
  └─ vite.config.js
```

```jsx
// src/main.jsx — 진입점
import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

ReactDOM.createRoot(document.getElementById('root')).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)
```

```jsx
// src/App.jsx — 루트 컴포넌트
import { useState } from 'react'

function App() {
  const [count, setCount] = useState(0)

  return (
    <div>
      <h1>첫 번째 React 앱</h1>
      <p>카운트: {count}</p>
      <button onClick={() => setCount(count + 1)}>
        클릭!
      </button>
    </div>
  )
}

export default App
```

---

## 13. 핵심 요약

```
React.js 1강 핵심:

개발 환경:
  WSL(Ubuntu 22.04) + Node.js + npm
  Vite로 프로젝트 생성 (Webpack 대신 현재 표준)
  package.json: 프로젝트 설정 + 의존성 관리

핵심 개념:
  SPA:          페이지 전환 시 JS로 화면 교체 (깜빡임 없음)
  JSX:          JS 안에 HTML처럼 쓰는 문법 → Babel이 변환
  Component:    UI의 독립적 단위, Props + State + JSX + Side Effect
  Virtual DOM:  변경된 부분만 Real DOM에 반영 → 효율적 업데이트
  Render:       state/props 변경 시 컴포넌트 함수 재실행

컴포넌트 작성:
  Class 컴포넌트 → 구형 (레거시 이해용)
  Function 컴포넌트 + Hooks → 현재 표준

주요 Hooks:
  useState      → state 관리 (변경 시 리렌더링)
  useEffect     → 사이드 이펙트 (API 호출, 구독 등)
  useRef        → DOM 참조 / 렌더링 없는 값 저장
  useContext    → 전역 상태 공유 (Prop Drilling 방지)
  useReducer    → 복잡한 상태 로직 (Redux 패턴)

useContext + useReducer 결합 → 경량 전역 상태 관리
```