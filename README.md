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

## 문서 업로드에서 할 일 후보까지

Spring은 업로드 커밋 후 `POST /internal/documents/{documentId}/analyze`를 호출합니다. `X-Internal-Token`과 경로·본문의 문서 ID 일치 여부를 검사하고 202로 접수합니다. 본문의 `objectKey`와 `mimeType`은 Spring 요청 계약에 따라 받지만 현재 무시합니다. 본문 값과 DB 값의 차이도 검사하지 않으며, 백그라운드 작업은 DB에 저장된 경로와 MIME 유형을 사용합니다.

같은 분석이 이미 RUNNING 또는 SUCCEEDED이면 중복 실행하지 않습니다. 워커가 분석 중 비정상 종료되면 RUNNING/PROCESSING 상태가 남을 수 있으며 같은 작업의 자동 재시작은 현재 지원하지 않습니다.

추출된 SCHEDULE, MEDICATION, CAUTION 항목 중 페이지 원문에 근거 문장이 있는 항목을 승인 대기 후보로 저장합니다. 후보와 분석 완료 상태를 같은 트랜잭션으로 저장하며 이 단계에서는 일정이나 예약을 만들지 않습니다. 날짜·시간은 원문의 명시된 값과 일치할 때만 사용하고, 연도·시간이 없거나 모호하면 보호자가 입력하도록 null로 남깁니다. 시간대가 없는 명시적 한국 시각은 Asia/Seoul로 처리합니다. 복약 빈도를 임의의 반복 일정으로 펼치지 않습니다.

일정 후보를 검사와 재방문으로 나누는 규칙은 원문 키워드를 사용합니다. 두 내용이 한 문장에 함께 있으면 재방문이 검사로 분류될 수 있습니다.

문서와 항목 내용으로 안정적인 후보 ID를 만들어 같은 항목의 재처리에도 보호자의 승인·거절 결정을 유지합니다. 성공한 재분석에서 사라진 승인 대기 후보는 취소하며, 동일 후보가 다시 추출되면 같은 ID로 승인 대기 상태에 복구합니다. 원본 문서가 삭제되면 Spring의 기존 이벤트가 승인 대기 후보를 취소합니다.

일반 실행은 `.env`를 설정하고 `python -m uvicorn app.main:app --host 127.0.0.1 --port 8000`으로 합니다. Spring과 같은 DATABASE_URL, STORAGE_LOCAL_ROOT, AI_INTERNAL_TOKEN을 사용해야 하며 문서 판독에는 OPENAI_API_KEY와 계정에서 지원하는 OPENAI_VLM_MODEL이 필요합니다. API 키가 없으면 실제 판독은 실패로 기록하며 가짜 성공 결과를 반환하지 않습니다.

키 없이 연결만 검증할 때는 `python -m tests.manual_contract_server`를 명시적으로 실행합니다. 이 테스트 진입점은 VLM 응답만 고정된 예시로 대체하고 파일 읽기·페이지 저장·결과 검증·후보 DB 저장은 실제 경로를 실행합니다. 프론트엔드는 VITE_AI_TEST_MODE=true로 예시 응답임을 표시해야 합니다. 이 모드는 AI 판독 품질이나 실제 공급자 호출을 검증하지 않으며 배포 진입점으로 사용하지 않습니다.

## 예방접종 일정 계산

Spring은 `POST /internal/vaccinations/schedule`(`X-Internal-Token`)에 `birthDate`, `asOf`, `completedDoses`를 보내 질병관리청 2026 일정 기준 묶음(now/soon/past)·다음 접종·개수를 받습니다. DB를 쓰지 않고 `app/domain/vaccination/data/kdca_2026-01-02.json`만 사용합니다.
