---
title: 056 Component Lifecycle & Hooks
tag:
  - 헬스케어 ai
  - react
description: 260722 수업 내용 정리
---

# Component Lifecycle & Hooks

React 컴포넌트는 화면에 나타나고, 데이터가 바뀌어 업데이트되고, 사라지는 세 가지 단계를 거친다. 이것을 **생명주기(Lifecycle)** 라고 한다. Class 컴포넌트 시절에는 이 단계를 별도의 메서드로 처리했고, Function 컴포넌트에서는 Hooks가 동일한 역할을 한다. 두 방식의 대응 관계를 이해하면 레거시 코드를 읽는 것과 현대적인 코드를 작성하는 것 모두 수월해진다.

---

## 1. 컴포넌트 생명주기 (Component Lifecycle)

컴포넌트의 생명주기는 크게 세 단계로 나뉜다. **Mounting**(태어남) → **Updating**(변화) → **Unmounting**(사라짐)이다.

```
시간 흐름 →

Mounting           Updating                    Unmounting
(컴포넌트 생성)    (props/state 변경)           (컴포넌트 제거)

constructor        new props / setState /       componentWillUnmount
    ↓              forceUpdate
render             ↓
    ↓              getDerivedStateFromProps
componentDidMount  ↓
                   shouldComponentUpdate
                   ↓
                   render
                   ↓
                   componentDidUpdate
```

### Render Phase vs Commit Phase

React는 컴포넌트를 화면에 그리는 과정을 두 단계로 나눈다.

**Render Phase**: 컴포넌트 함수(또는 `render()` 메서드)를 실행해서 새로운 Virtual DOM을 만드는 단계다. 이 단계는 순수해야 하며 부수 효과(Side Effect)가 없어야 한다. React가 중단했다가 다시 시작할 수도 있다.

**Commit Phase**: Render Phase에서 만든 Virtual DOM을 실제 Real DOM에 반영하는 단계다. `componentDidMount`, `componentDidUpdate` 같은 생명주기 메서드가 이 단계에서 실행된다.

```
Render Phase:   constructor → getDerivedStateFromProps → render
                → 순수 계산만 수행, Side Effect 없어야 함

Commit Phase:   componentDidMount / componentDidUpdate
                → Real DOM 반영 후 실행
                → API 호출, 구독 설정, DOM 직접 조작 등 허용
```

---

## 2. Class 컴포넌트 생명주기 메서드

### Mounting 단계

**constructor(props)**: 컴포넌트가 생성될 때 가장 먼저 호출된다. state 초기화와 이벤트 핸들러 바인딩에 사용한다. `super(props)` 호출이 필수다.

**getDerivedStateFromProps(props, state)**: props가 바뀔 때 state를 동기화해야 하는 경우에 사용한다. 정적 메서드이며 사용 빈도가 낮다.

**render()**: JSX를 반환하는 필수 메서드다. 순수 함수여야 하며, 이 안에서 setState를 호출하면 안 된다.

**componentDidMount()**: 컴포넌트가 Real DOM에 처음 추가된 직후 실행된다. API 데이터 로드, 구독 설정, 타이머 시작 등 Side Effect가 여기서 이루어진다.

### Updating 단계

**shouldComponentUpdate(nextProps, nextState)**: 리렌더링 여부를 결정한다. false를 반환하면 렌더링을 건너뛴다. 성능 최적화에 사용한다.

**componentDidUpdate(prevProps, prevState)**: 업데이트 후 Real DOM에 반영된 뒤 실행된다. 이전 props/state와 비교해서 추가 작업을 할 때 사용한다.

### Unmounting 단계

**componentWillUnmount()**: 컴포넌트가 제거되기 직전에 실행된다. 이벤트 리스너 제거, 타이머 정리, 구독 해제 등 정리 작업이 여기서 이루어진다.

```jsx
import React, { Component } from 'react';

class LifecycleDemo extends Component {
  constructor(props) {
    super(props);
    this.state = { data: null, count: 0 };
    console.log('1. constructor — 컴포넌트 생성');
  }

  static getDerivedStateFromProps(props, state) {
    // props에서 state 동기화 (드물게 사용)
    return null;
  }

  componentDidMount() {
    console.log('3. componentDidMount — DOM에 추가됨');
    // API 호출, 타이머 시작, 구독 등
    this.timer = setInterval(() => {
      this.setState(prev => ({ count: prev.count + 1 }));
    }, 1000);
  }

  shouldComponentUpdate(nextProps, nextState) {
    // count가 10 이상이면 리렌더링 중단
    return nextState.count < 10;
  }

  componentDidUpdate(prevProps, prevState) {
    console.log('5. componentDidUpdate — 업데이트됨');
    if (prevState.count !== this.state.count) {
      console.log(`count: ${prevState.count} → ${this.state.count}`);
    }
  }

  componentWillUnmount() {
    console.log('6. componentWillUnmount — 제거 직전');
    clearInterval(this.timer);   // 타이머 정리 (메모리 누수 방지)
  }

  render() {
    console.log('2/4. render — Virtual DOM 생성');
    return <div>Count: {this.state.count}</div>;
  }
}
```

---

## 3. Hooks의 생태계 — 3대 기능적 구역

Function 컴포넌트에서 Hooks는 역할에 따라 세 그룹으로 나뉜다.

```
Blueprint Map (Hook 생태계):

┌─────────────────┐  ┌──────────────────┐  ┌──────────────────────┐
│  상태 관리      │  │  사이드 이펙트   │  │  성능 최적화 및 참조  │
│ (State Mgmt)    │  │  (Side Effects)  │  │  (Optimization & Ref)│
│                 │  │                  │  │                      │
│  useState       │  │  useEffect       │  │  useMemo             │
│  useReducer     │  │                  │  │  useCallback         │
│  useContext     │  │                  │  │  useRef              │
└─────────────────┘  └──────────────────┘  └──────────────────────┘
```

### 그룹 1 — 상태 관리 (State Management)

**useState**: 단순한 값 하나의 상태 관리. 변경 시 컴포넌트 리렌더링 발생.

**useReducer**: 여러 상태가 복잡하게 얽혀 있거나 상태 변환 로직이 복잡할 때. Redux 패턴과 동일.

**useContext**: 컴포넌트 트리 깊이와 상관없이 전역 데이터 공유.

### 그룹 2 — 사이드 이펙트 (Side Effects)

**useEffect**: API 호출, 이벤트 리스너, 타이머, DOM 직접 조작 등 렌더링 외부의 작업.

### 그룹 3 — 성능 최적화 및 참조 (Optimization & Refs)

**useMemo**: 계산량이 많은 값을 메모이제이션. 의존성이 바뀔 때만 재계산.

**useCallback**: 함수를 메모이제이션. 자식 컴포넌트에 props로 함수를 전달할 때 불필요한 리렌더링 방지.

**useRef**: DOM 참조 또는 렌더링을 유발하지 않는 값 저장.

---

## 4. useEffect 상세

Class 컴포넌트의 `componentDidMount`, `componentDidUpdate`, `componentWillUnmount`를 **하나의 Hook**으로 통합한 것이 `useEffect`다.

### useEffect의 세 가지 형태

```jsx
// 형태 1: 의존성 배열 없음 — 매 렌더링 후 실행
useEffect(() => {
  console.log('렌더링될 때마다 실행');
});

// 형태 2: 빈 배열 — 마운트 시 한 번만 실행 (componentDidMount 대응)
useEffect(() => {
  console.log('마운트 시 한 번만 실행');
  // 구독, API 초기 로드 등
}, []);

// 형태 3: 의존성 배열 — 특정 값이 바뀔 때만 실행 (componentDidUpdate 대응)
useEffect(() => {
  console.log('count가 바뀔 때만 실행:', count);
}, [count]);
```

### 정리 함수 (Cleanup) — componentWillUnmount 대응

`useEffect` 안에서 함수를 반환하면 그것이 정리 함수가 된다. 컴포넌트가 언마운트되거나, 다음 Effect가 실행되기 전에 호출된다.

```jsx
useEffect(() => {
  // 구독 시작
  const subscription = someAPI.subscribe(userId, handleUpdate);
  const timerId = setInterval(fetchData, 5000);

  // 정리 함수: 컴포넌트 제거 시 또는 userId 변경 시 실행
  return () => {
    subscription.unsubscribe();   // 구독 해제
    clearInterval(timerId);        // 타이머 정리
  };
}, [userId]);   // userId가 바뀔 때마다 정리 → 재구독
```

### Class 생명주기와 useEffect 대응표

|Class 메서드|useEffect 패턴|
|---|---|
|`componentDidMount`|`useEffect(() => {...}, [])`|
|`componentDidUpdate`|`useEffect(() => {...}, [dep])`|
|`componentWillUnmount`|`useEffect(() => { return () => {...} }, [])`|
|componentDidMount + Update|`useEffect(() => {...})`|

```jsx
// 완전한 useEffect 활용 예시
function DataFetcher({ userId }) {
  const [data, setData]       = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError]     = useState(null);

  useEffect(() => {
    let cancelled = false;   // 비동기 중복 요청 방지

    setLoading(true);
    setError(null);

    fetch(`https://api.example.com/users/${userId}`)
      .then(res => {
        if (!res.ok) throw new Error('서버 오류');
        return res.json();
      })
      .then(data => {
        if (!cancelled) {   // 컴포넌트가 이미 언마운트됐으면 무시
          setData(data);
          setLoading(false);
        }
      })
      .catch(err => {
        if (!cancelled) {
          setError(err.message);
          setLoading(false);
        }
      });

    return () => { cancelled = true; };   // 정리: 요청 취소 플래그

  }, [userId]);   // userId가 바뀔 때마다 새로 요청

  if (loading) return <p>로딩 중...</p>;
  if (error)   return <p>오류: {error}</p>;
  return <div>{JSON.stringify(data)}</div>;
}
```

---

## 5. Events Handler와 Virtual DOM

### 이벤트 흐름

```
App                React                  DOM
 │                  │                      │
 │   사용자 상호작용 │                      │
 │←─────────────────────── Delivers events ─┤
 │                  │                      │
 │ setState 호출    │                      │
 ├──────────────→   │                      │
 │                  │ Builds/Modifies      │
 │            Virtual DOM 업데이트         │
 │                  ├──────────────────→   │
 │                  │  Builds/Modifies     │
 │                  │    Real DOM          │
 │                  │                      │
 │←─────────────────────── Delivers events ─┤
 │              (React 이벤트 위임)
```

React는 모든 이벤트를 **루트 DOM 노드 하나에 위임(Event Delegation)** 해서 처리한다. 각 DOM 요소에 직접 이벤트 리스너를 붙이지 않고, 루트에서 버블링된 이벤트를 받아 적절한 컴포넌트에 전달한다. 이 덕분에 메모리 효율이 높고, 동적으로 추가되는 요소에도 자동으로 이벤트가 적용된다.

### 이벤트 핸들러 작성 패턴

```jsx
function EventDemo() {
  const [text, setText] = useState('');

  // 패턴 1: 인라인 화살표 함수 (간단한 경우)
  return (
    <button onClick={() => console.log('클릭!')}>
      클릭
    </button>
  );

  // 패턴 2: 핸들러 함수 분리 (복잡한 로직)
  const handleClick = (e) => {
    e.preventDefault();   // 기본 동작 방지
    e.stopPropagation(); // 이벤트 버블링 방지
    console.log('버튼 클릭:', e.target);
  };

  // 패턴 3: 인자가 있는 핸들러
  const handleItemClick = (id) => (e) => {
    console.log('아이템 클릭:', id);
  };

  return (
    <div>
      <button onClick={handleClick}>클릭</button>
      <input
        value={text}
        onChange={(e) => setText(e.target.value)}  // Controlled Component
        onKeyDown={(e) => e.key === 'Enter' && handleSubmit()}
      />
      {items.map(item => (
        <div key={item.id} onClick={handleItemClick(item.id)}>
          {item.name}
        </div>
      ))}
    </div>
  );
}
```

---

## 6. 성능 최적화 Hooks — useMemo, useCallback, useRef

### useMemo — 값 메모이제이션

계산 비용이 높은 값을 캐싱한다. 의존성이 바뀌지 않으면 이전 결과를 그대로 반환한다.

```jsx
import { useState, useMemo } from 'react';

function ExpensiveList({ items, filter }) {
  // filter가 바뀔 때만 재계산 (items가 수만 개여도 filter 변경 시만 연산)
  const filteredItems = useMemo(() => {
    console.log('필터링 중...');
    return items.filter(item => item.name.includes(filter));
  }, [items, filter]);   // items나 filter가 바뀔 때만 재계산

  return (
    <ul>
      {filteredItems.map(item => <li key={item.id}>{item.name}</li>)}
    </ul>
  );
}
```

언제 쓰는가:

- 배열 필터링, 정렬처럼 O(n) 이상의 연산
- 렌더링마다 반복되는 복잡한 계산
- 자식 컴포넌트에 전달하는 객체/배열 (참조 동일성 보장)

### useCallback — 함수 메모이제이션

함수를 캐싱한다. 자식 컴포넌트에 함수를 props로 전달할 때, 부모가 리렌더링되어도 같은 함수 참조가 유지되어 자식의 불필요한 리렌더링을 방지한다.

```jsx
import { useState, useCallback, memo } from 'react';

// React.memo: props가 바뀌지 않으면 리렌더링 안 함
const ChildButton = memo(({ onClick, label }) => {
  console.log('ChildButton 렌더링:', label);
  return <button onClick={onClick}>{label}</button>;
});

function Parent() {
  const [count, setCount] = useState(0);
  const [text, setText]   = useState('');

  // useCallback 없으면: Parent 리렌더링 시마다 새 함수 생성
  // → ChildButton이 memo여도 항상 리렌더링됨

  // useCallback 사용: 함수 참조 유지
  // → ChildButton은 onClick이 실제로 바뀔 때만 리렌더링됨
  const handleIncrement = useCallback(() => {
    setCount(c => c + 1);
  }, []);   // 의존성 없음 → 항상 같은 함수 참조

  return (
    <div>
      <p>Count: {count}</p>
      <input value={text} onChange={e => setText(e.target.value)} />
      <ChildButton onClick={handleIncrement} label="증가" />
    </div>
  );
}
```

### useRef — DOM 참조와 값 저장

`useRef`는 두 가지 용도로 사용한다.

```jsx
import { useRef, useEffect } from 'react';

function RefDemo() {
  // 용도 1: DOM 요소 직접 참조
  const inputRef = useRef(null);

  useEffect(() => {
    inputRef.current.focus();   // 마운트 시 자동 포커스
  }, []);

  // 용도 2: 렌더링을 유발하지 않는 값 저장
  // (useState와 달리 값이 바뀌어도 리렌더링 없음)
  const renderCount = useRef(0);
  const prevCountRef = useRef();
  const timerId = useRef(null);

  useEffect(() => {
    renderCount.current += 1;   // 렌더링 횟수 추적 (리렌더링 없음)
    prevCountRef.current = count;   // 이전 값 저장
  });

  const startTimer = () => {
    timerId.current = setInterval(() => {...}, 1000);
  };
  const stopTimer = () => {
    clearInterval(timerId.current);   // ref로 저장된 타이머 ID 사용
  };

  return (
    <div>
      <input ref={inputRef} placeholder="자동 포커스" />
      <p>이전 count: {prevCountRef.current}</p>
    </div>
  );
}
```

### useMemo vs useCallback 비교

|Hook|캐싱 대상|사용 시점|
|---|---|---|
|`useMemo`|**계산된 값**|비용 높은 연산 결과 재사용|
|`useCallback`|**함수**|자식 컴포넌트에 함수 props 전달 시|

```
useMemo:     const value = useMemo(() => compute(), [dep])
useCallback: const fn    = useCallback(() => {...}, [dep])

useMemo(() => fn, [dep])  ≡  useCallback(fn, [dep])
← 실제로 useCallback은 useMemo의 특수한 형태
```

---

## 7. Props Drilling 문제와 해결

### Props Drilling이란

컴포넌트 트리가 깊어지면, 중간 단계의 컴포넌트들이 데이터를 직접 사용하지 않으면서도 자식에게 전달하기 위해 props를 받아야 하는 문제가 발생한다.

```
Props Drilling 문제:

App (state: user)
  ↓ props: user
  Layout
    ↓ props: user (사용 안 함, 그냥 전달만)
    Sidebar
      ↓ props: user (사용 안 함, 그냥 전달만)
      UserCard
        ↓ props: user
        Avatar (user를 실제로 사용)

→ Layout, Sidebar는 user를 사용하지 않지만 전달을 위해 props를 받아야 함
→ user가 바뀌면 중간 컴포넌트들도 모두 리렌더링됨
→ 코드가 복잡해지고 유지보수가 어려워짐
```

### 해결책 1 — useContext

```jsx
import { createContext, useContext, useState } from 'react';

// 1. Context 생성
const UserContext = createContext(null);

// 2. 최상위에서 Provider로 공급
function App() {
  const [user, setUser] = useState({ name: '수용', role: 'admin' });

  return (
    <UserContext.Provider value={{ user, setUser }}>
      <Layout />   {/* props 전달 없음 */}
    </UserContext.Provider>
  );
}

// 3. 필요한 곳에서 바로 소비
function Avatar() {
  const { user } = useContext(UserContext);   // 직접 접근
  return <img alt={user.name} />;
}
```

### 해결책 2 — useContext + useReducer (전역 상태 관리)

```jsx
import { createContext, useContext, useReducer } from 'react';

// Reducer 정의
function userReducer(state, action) {
  switch (action.type) {
    case 'UPDATE_NAME':  return { ...state, name: action.payload };
    case 'UPDATE_ROLE':  return { ...state, role: action.payload };
    case 'LOGOUT':       return null;
    default:             return state;
  }
}

// Context 생성
const UserStateContext    = createContext(null);
const UserDispatchContext = createContext(null);

// Provider 컴포넌트
function UserProvider({ children }) {
  const [user, dispatch] = useReducer(userReducer, { name: '수용', role: 'user' });

  return (
    <UserStateContext.Provider value={user}>
      <UserDispatchContext.Provider value={dispatch}>
        {children}
      </UserDispatchContext.Provider>
    </UserStateContext.Provider>
  );
}

// 커스텀 Hook으로 사용 편의성 향상
function useUser()         { return useContext(UserStateContext); }
function useUserDispatch() { return useContext(UserDispatchContext); }

// 사용
function ProfileHeader() {
  const user = useUser();
  const dispatch = useUserDispatch();

  return (
    <div>
      <h2>{user.name}</h2>
      <button onClick={() => dispatch({ type: 'LOGOUT' })}>로그아웃</button>
    </div>
  );
}
```

---

## 8. Function 컴포넌트 생명주기 — useEffect로 완전 대체

Class 컴포넌트의 생명주기 전체를 Function 컴포넌트에서 `useEffect`로 구현하면 다음과 같다.

```jsx
import { useState, useEffect, useRef } from 'react';

function FullLifecycleDemo({ userId }) {
  const [data, setData] = useState(null);
  const isMounted = useRef(true);

  // componentDidMount: 마운트 시 한 번 실행
  useEffect(() => {
    console.log('마운트됨');
    return () => {
      isMounted.current = false;
      console.log('언마운트됨 — componentWillUnmount');
    };
  }, []);

  // componentDidUpdate (userId 변경 감지)
  useEffect(() => {
    if (!userId) return;
    console.log('userId 변경됨:', userId);

    fetch(`/api/user/${userId}`)
      .then(r => r.json())
      .then(d => {
        if (isMounted.current) setData(d);
      });
  }, [userId]);

  return <div>{data?.name}</div>;
}
```

### Class vs Function 생명주기 최종 비교

|시점|Class 컴포넌트|Function 컴포넌트 (Hooks)|
|---|---|---|
|생성/초기화|`constructor`|`useState` 초기값, `useRef` 초기값|
|첫 렌더링 후|`componentDidMount`|`useEffect(() => {...}, [])`|
|업데이트 후|`componentDidUpdate`|`useEffect(() => {...}, [dep])`|
|제거 직전|`componentWillUnmount`|`useEffect(() => { return () => {...} }, [])`|
|렌더링 최적화|`shouldComponentUpdate`|`React.memo`, `useMemo`, `useCallback`|
|에러 처리|`componentDidCatch`|`ErrorBoundary` (Class만 지원)|

---

## 9. 핵심 요약

```
컴포넌트 생명주기 3단계:
  Mounting   → 생성: constructor → render → componentDidMount
  Updating   → 업데이트: getDerivedState → shouldUpdate → render → componentDidUpdate
  Unmounting → 제거: componentWillUnmount

실행 단계 2단계:
  Render Phase  → Virtual DOM 계산 (순수, Side Effect 없어야 함)
  Commit Phase  → Real DOM 반영 후 생명주기 메서드 실행

Hook 3대 구역:
  상태 관리:       useState, useReducer, useContext
  사이드 이펙트:   useEffect
  최적화 & 참조:   useMemo, useCallback, useRef

useEffect 형태 3가지:
  useEffect(() => {...})           → 매 렌더링 후
  useEffect(() => {...}, [])       → 마운트 시 1회
  useEffect(() => {...}, [a, b])   → a 또는 b 변경 시

Props Drilling 해결:
  useContext                       → 전역 데이터 공유
  useContext + useReducer          → 경량 전역 상태 관리 (Redux 대안)

성능 최적화:
  useMemo      → 계산값 캐싱 (의존성 변경 시만 재계산)
  useCallback  → 함수 캐싱 (자식 컴포넌트 불필요한 리렌더링 방지)
  React.memo   → 컴포넌트 메모이제이션 (props 미변경 시 리렌더링 건너뜀)
```