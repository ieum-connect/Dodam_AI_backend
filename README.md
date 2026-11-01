# 도담(DODAM) AI 백엔드

도담 서비스의 AI 백엔드(FastAPI)입니다. 업로드된 의료 문서 분석과 쉬운 설명, 기록 기반 상담 응답, 복지 추천 근거 생성을 담당합니다.

## 실행

```bash
uv sync            # 또는 pip install -e .
uvicorn app.main:app --reload
```

## 검증

```bash
ruff check .
pytest
```

---

## 📚 팀 개발 컨벤션

도담(DODAM) 프로젝트는 프론트엔드, Spring 백엔드, AI 백엔드가 공통 개발 컨벤션을 따릅니다.

- [브랜치 컨벤션](https://github.com/ieum-connect/Dodam_backend/blob/develop/docs/convention/BRANCH_CONVENTION.md)
- [커밋 컨벤션](https://github.com/ieum-connect/Dodam_backend/blob/develop/docs/convention/COMMIT_CONVENTION.md)
- [PR 컨벤션](https://github.com/ieum-connect/Dodam_backend/blob/develop/docs/convention/PR_CONVENTION.md)

개발 절차: Issue 생성 → 브랜치 생성 → 개발 및 커밋 → PR 생성 → 코드 리뷰 → 승인 → 병합
