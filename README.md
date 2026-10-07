# yun-sooyong.github.io

윤수용의 포트폴리오 사이트. 빌드 단계 없는 정적 페이지(HTML · CSS · JS)이고 `main`에 푸시하면 GitHub Actions로 배포된다.

- `/` : 포트폴리오 (이 레포)
- `/blog/` : 학습 노트 블로그 ([Yun-sooyong/blog](https://github.com/Yun-sooyong/blog), Quartz)

## 프로젝트 데이터
`data/projects.json`은 비공개 허브 레포의 `projects.yaml`에서 생성한다. 직접 고치지 않는다.

```bash
# portfolio-hub 레포에서
python scripts/export_site_data.py ../yun-sooyong.github.io
```

`status`가 `active`/`done`이고 한 줄 소개가 확인된 프로젝트만 나가며, `[확인 필요]` 값과 내부 메모는 빠진다.

## 로컬 미리보기
```bash
python -m http.server 8000
```

## 기타
- `404.html` : 블로그를 `/blog`로 옮기기 전 주소(`/글이름`)로 들어오면 `/blog/글이름`으로 보낸다.
