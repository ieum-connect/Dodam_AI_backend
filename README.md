<div align="center">

<img src="./docs/assets/dodam_logo_effect.gif" width="100%" alt="도담 애니메이션 배너" />

# 도담 AI Backend

![Python](https://img.shields.io/badge/Python_3.12-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![pgvector](https://img.shields.io/badge/pgvector-4169E1?style=flat-square&logo=postgresql&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI_API-412991?style=flat-square&logo=openai&logoColor=white)
![Pytest](https://img.shields.io/badge/Pytest-0A9EDC?style=flat-square&logo=pytest&logoColor=white)

[Frontend](https://github.com/ieum-connect/Dodam_frontend) · [Spring Backend](https://github.com/ieum-connect/Dodam_backend)

</div>

---

## 1. ✨ 핵심 기능

| 기능 | 설명 |
|---|---|
| 📄 의료 문서 분석 | PDF·JPG·PNG 문서를 페이지 이미지로 변환하고 VLM으로 원문과 구조화 항목을 추출합니다. |
| 🧾 근거 위치 보존 | 추출 항목에 페이지 번호, 원문과 좌표를 연결하여 결과의 출처를 추적할 수 있게 합니다. |
| 📅 할 일 후보 생성 | 문서에 명시된 일정·복약·주의사항만 보호자 승인 대기 후보로 생성합니다. |
| 🧩 문서 청킹 | 페이지 원문을 최대 500 Token의 겹치는 Chunk로 분리합니다. |
| 🔎 의료 RAG 검색 | 질문을 Embedding하고 허용된 아이·문서의 Chunk만 pgvector로 검색합니다. |
| 💬 근거 기반 답변 | 검색 근거를 사용해 쉬운 설명·비교·복약 요약을 생성하고 Citation을 검증합니다. |
| 🛡️ 의료 안전 차단 | 진단, 약 변경, 치료 결정과 응급 판단 요청은 LLM 호출 전에 차단합니다. |
| 🎁 복지 설명 | Spring 추천 엔진의 판정과 원문을 바탕으로 추천 이유를 설명합니다. |
| 💉 예방접종 일정 | 생년월일·기준일·완료 차수를 질병관리청 일정 Data와 비교해 현재·예정·지난 접종을 계산합니다. |
| ☁️ 저장소 전환 | 개발 환경은 Local Folder, 배포 환경은 비공개 S3에서 같은 문서 Pipeline을 실행합니다. |
| ♻️ 재색인 | VLM을 다시 호출하지 않고 저장된 원문으로 Chunk와 Embedding을 복구합니다. |

<details open>
<summary><b>💡 AI 용어를 쉽게 이해하기</b></summary>

| 용어 | 쉬운 설명 |
|---|---|
| VLM | 문서 이미지를 보고 글자와 표의 내용을 읽는 AI입니다. 사람이 검사 결과지를 눈으로 읽어 옮기는 단계와 비슷합니다. |
| Chunk | 긴 문서 원문을 검색하기 좋은 작은 글 묶음으로 나눈 것입니다. 도담은 Page를 섞지 않고 문장과 표의 행을 최대한 보존합니다. |
| Embedding | 글의 의미를 숫자 좌표로 바꾼 값입니다. 질문과 의미가 가까운 문장 조각을 찾는 데 사용합니다. |
| Vector Search | 질문의 숫자 좌표와 문서 조각의 좌표를 비교하여 의미가 가까운 내용을 찾는 검색 방식입니다. |
| RAG | 먼저 관련 원문을 검색하고, 그 원문만 근거로 AI가 답변하도록 만드는 방식입니다. |
| Citation | AI 답변이 어느 문서의 어느 원문에서 나왔는지 가리키는 근거 ID입니다. |
| Grounding | AI가 추측하지 않고 제공된 문서 원문 안에서만 답하도록 제한하는 원칙입니다. |
| 재색인 | 이미 읽어둔 문서 원문으로 검색용 Chunk와 Embedding만 다시 만드는 복구 작업입니다. |

</details>

<details>
<summary><b>🧠 AI 백엔드가 하지 않는 일</b></summary>

- 사용자의 로그인과 보호자 권한을 직접 관리하지 않습니다.
- Frontend의 요청을 직접 받지 않습니다.
- 의료 진단, 처방 변경, 치료 선택 또는 위험도를 확정하지 않습니다.
- 문서에 없는 날짜·수치·단위·소아 참고 범위를 추측하지 않습니다.
- 복지 대상 여부와 추천 점수를 새로 계산하지 않습니다.
- 추출된 일정과 복약 정보를 보호자 승인 없이 실제 일정으로 등록하지 않습니다.

</details>

---

## 2. 🧭 전체 시스템 아키텍처

```mermaid
flowchart LR
    USER["👨‍👩‍👧 보호자"] --> FE["🎨 React PWA"]
    FE -->|Cookie 인증 · /api/*| BE

    subgraph SPRING["⚙️ Spring Backend"]
        BE["인증 · 권한 · 비즈니스 로직"]
        UPLOAD["문서 업로드 · 상태 관리"]
        CHAT["AI 상담 Session · Message"]
        WENGINE["복지 추천 Engine"]
        BE --> UPLOAD
        BE --> CHAT
        BE --> WENGINE
    end

    subgraph AI["🤖 AI Backend · FastAPI"]
        API["Internal API"]
        VLM["Document VLM Pipeline"]
        INDEX["Chunk · Embedding"]
        RAG["Medical RAG"]
        ANSWER["Grounded Answer"]
        WELFARE["Welfare Explanation"]
        VACCINE["Vaccination Schedule Engine"]
        API --> VLM --> INDEX --> RAG --> ANSWER
        API --> WELFARE
        API --> VACCINE
    end

    BE -->|X-Internal-Token| API
    WELFARE -->|복지 Catalog 조회| BE
    UPLOAD --> STORAGE[("📁 Shared Storage<br/>Local · Private S3")]
    VLM --> STORAGE
    BE --> DB[("🗄️ PostgreSQL · pgvector")]
    VLM --> DB
    INDEX --> DB
    RAG --> DB
    WELFARE --> DB
    VLM --> MODELS["☁️ VLM API"]
    INDEX --> MODELS
    ANSWER --> MODELS
    WELFARE --> MODELS
```

<details>
<summary><b>🔗 서비스 간 호출 원칙</b></summary>

1. Frontend는 Spring Backend만 호출합니다.
2. Spring Backend가 사용자 인증과 아이·문서 접근 권한을 검증합니다.
3. AI Backend 호출에는 양쪽 서버에 동일하게 설정한 `X-Internal-Token`을 사용합니다.
4. 의료 RAG 검색 시 Spring이 허용한 `patient_id`와 `allowed_document_ids`를 전달합니다.
5. AI Backend는 DB 조건을 다시 적용하여 삭제되지 않고 Embedding이 준비된 문서만 검색합니다.
6. 문서 원본은 Local Mode에서는 같은 Storage Root를, S3 Mode에서는 같은 비공개 Bucket과 Region을 사용합니다. Database도 같은 PostgreSQL을 사용합니다.

</details>

---

## 3. 🔄 핵심 AI 플로우

<details open>
<summary><b>📄 의료 문서 분석 Pipeline</b></summary>

> 한마디로, **업로드한 문서를 Page별로 읽고 → 확인 가능한 정보만 저장하고 → 검색 가능한 형태로 준비하는 과정**입니다.

```mermaid
flowchart TD
    A["Spring: 문서 업로드 완료"] --> B["POST /internal/documents/{id}/analyze"]
    B --> C["202 Accepted · Background Task"]
    C --> D{"분석 대상인가?"}
    D -->|삭제 · 완료 · 대상 없음| SKIP["변경 없이 종료"]
    D -->|분석 가능| START["PROCESSING 상태 Commit"]
    START --> LOAD["DB의 Storage Key로 원본 읽기"]
    LOAD --> SPLIT["PDF / Image → Page PNG"]
    SPLIT --> SAVE["Rendered Page 저장"]
    SAVE --> EXTRACT["Page별 VLM 구조화 추출"]
    EXTRACT --> RETRY{"일시적 오류?"}
    RETRY -->|Timeout · Rate Limit · API Error| EXTRACT
    RETRY -->|완료| VALIDATE["Strict Schema · 주민번호 Masking"]
    VALIDATE --> RESULT{"Page 성공 범위"}
    RESULT -->|전체 성공| READY["Document READY"]
    RESULT -->|일부 성공| REVIEW["Document NEEDS_REVIEW"]
    RESULT -->|전체 실패| FAILED["Document FAILED"]
    READY --> CANDIDATE["할 일 후보 생성"]
    REVIEW --> CANDIDATE
    CANDIDATE --> CHUNK["원문 Chunking"]
    CHUNK --> EMBED["Embedding 생성"]
    EMBED --> VECTOR["Chunk · Vector 교체 저장"]
```

#### 문서 처리 규칙

- 지원 형식: `application/pdf`, `image/jpeg`, `image/png`
- 기본 Page 제한: 30 Page
- 기본 Rendering DPI: 150
- Page 최대 변 길이: 2048px
- VLM은 Timeout·Rate Limit·일시적 API 오류에만 지수 Backoff로 재시도합니다.
- VLM 호출은 오래 걸리므로 Database Transaction 밖에서 실행합니다.
- 분석 결과 저장이 끝난 뒤 Chunk와 Embedding을 생성합니다.
- 색인 실패는 추출 결과를 되돌리지 않으며 별도 재색인으로 복구합니다.

</details>

<details>
<summary><b>🔎 의료 문서 RAG 검색</b></summary>

> 질문과 의미가 가까운 문서 조각을 찾되, **현재 아이에게 속하고 Spring이 허용한 문서만** 검색합니다.

```mermaid
sequenceDiagram
    participant BE as ⚙️ Spring Backend
    participant AI as 🤖 AI Backend
    participant EMB as Embedding Provider
    participant DB as 🗄️ PostgreSQL · pgvector

    BE->>AI: question + patientId + allowedDocumentIds
    AI->>EMB: 질문 Embedding 생성
    EMB-->>AI: vector(768)
    AI->>DB: 허용된 문서 Chunk Vector 검색
    Note over AI,DB: 같은 Patient · 삭제되지 않음<br/>Embedding READY · Model/Dimension 일치
    DB-->>AI: 유사 Chunk · Page · 원문 · Similarity
    AI->>DB: 검색 가능 문서와 Index 상태 확인
    AI-->>BE: Evidence + Citation ID + Warning
```

검색 결과는 다음 상태를 함께 전달합니다.

- `PARTIAL_DOCUMENT_ANALYSIS`: 일부 Page만 분석된 문서
- `DOCUMENT_INDEX_PENDING`: Embedding 준비 중
- `DOCUMENT_INDEX_FAILED`: 색인 실패

</details>

<details>
<summary><b>💬 Citation 기반 답변 생성</b></summary>

> 검색된 원문을 바탕으로 답변을 만든 뒤, **근거가 확인되지 않는 문장은 마지막 단계에서 제거**합니다.

```mermaid
flowchart TD
    Q["보호자 질문 + 검색 Evidence"] --> SAFE{"의료 판단 요청인가?"}
    SAFE -->|진단 · 약 변경 · 치료 · 응급 판단| BLOCK["LLM 호출 없이 차단 안내"]
    SAFE -->|안전한 설명 요청| CLASSIFY["질문 유형 분류"]
    CLASSIFY -->|지원하지 않는 질문| UNSUPPORTED["질문 변경 안내"]
    CLASSIFY -->|설명 · 비교 · 복약 · 일정 · 출처| EVIDENCE{"Evidence가 있는가?"}
    EVIDENCE -->|없음| INSUFFICIENT["INSUFFICIENT"]
    EVIDENCE -->|있음| LLM["Grounded JSON 답변 생성"]
    LLM --> FILTER["허용된 Citation ID 검증"]
    FILTER -->|모든 문장 유효| COMPLETE["COMPLETE"]
    FILTER -->|일부 문장 제거| PARTIAL["PARTIAL"]
    FILTER -->|유효 문장 없음| INSUFFICIENT
```

LLM이 반환한 문장 중 Citation이 없거나 요청에서 허용하지 않은 Citation을 사용한 문장은 화면에 노출하지 않습니다.

</details>

<details>
<summary><b>🎁 복지 추천 설명</b></summary>

> 복지 대상 여부를 AI가 새로 판단하는 기능이 아닙니다. **Spring이 계산한 결과를 보호자가 이해하기 쉬운 말로 풀어주는 기능**입니다.

```mermaid
flowchart LR
    SPRING["Spring 추천 Engine<br/>점수 · 조건 판정"] --> REQUEST["Program · Recommendation<br/>Child Context · Evidence"]
    REQUEST --> AI["AI 설명 생성"]
    AI --> CHECK{"원문에 없는 % 생성?"}
    CHECK -->|아니요| RESULT["보호자용 추천 이유"]
    CHECK -->|예| FALLBACK["규칙 기반 설명으로 Fallback"]
    FAIL["LLM 오류"] --> FALLBACK
```

- 추천 점수와 조건 충족 여부는 Spring Backend가 결정합니다.
- AI Backend는 판정을 보호자가 이해하기 쉬운 문장으로 설명합니다.
- 입력에 없는 백분율을 생성하면 AI 결과를 폐기합니다.
- LLM 호출 실패 시 규칙 기반 설명을 반환합니다.
- 자격을 확정하지 않고 신청기관에서 최종 확인하도록 안내합니다.

</details>

---

## 4. 🛡️ 의료 안전과 Grounding

<details open>
<summary><b>🚧 LLM 호출 전 안전 장치</b></summary>

- 진단 요청, 약 중단·변경, 용량 변경, 치료 선택과 응급 판단 질문을 규칙으로 먼저 차단합니다.
- 호흡 곤란, 의식 저하, 경련, 청색증과 심한 출혈이 포함된 질문은 119 또는 응급실 안내를 우선합니다.
- 지원하지 않는 일반 질문은 의료 기록에서 확인 가능한 질문으로 변경하도록 안내합니다.
- 질문 유형 Hint도 의료 판단 차단을 우회할 수 없습니다.

</details>

<details>
<summary><b>📌 문서 추출 안전 규칙</b></summary>

- 문서에 보이는 내용만 추출하며 수치·단위·날짜를 추측하지 않습니다.
- 정상·비정상을 자체 판정하거나 성인 참고 범위를 소아에게 적용하지 않습니다.
- 문서 이미지 안에 포함된 지시문을 명령으로 따르지 않습니다.
- 주민등록번호와 외국인등록번호 뒷자리 7자를 `*`로 Masking합니다.
- Page 원문과 항목은 Strict JSON Schema로 검증합니다.
- 읽을 수 없는 값은 만들지 않고 `null`과 낮은 Confidence로 처리합니다.

</details>

<details>
<summary><b>🔐 API·Storage·Log 안전</b></summary>

- 내부 Endpoint는 `X-Internal-Token`으로 보호합니다.
- 분석 URL의 Document ID와 Body의 `documentId`가 다르면 거부합니다.
- 요청 Body의 `objectKey`를 파일 선택에 사용하지 않고 DB의 Storage Key를 읽습니다.
- Storage Root 밖을 가리키는 절대 경로와 경로 탈출을 거부합니다.
- 오류 Log에는 문서 원문, API Key와 실제 Storage 경로를 남기지 않습니다.
- 예상하지 못한 오류는 내부 정보를 숨긴 공통 JSON 오류로 반환합니다.

</details>

---

## 5. 🔌 내부 API

> `/health`와 복지 준비 상태를 제외한 기능 API는 Spring Backend의 내부 호출을 전제로 합니다.

<details open>
<summary><b>📄 문서 분석·RAG API</b></summary>

| Method | Endpoint | 설명 |
|---|---|---|
| `GET` | `/health` | FastAPI Process 상태 확인 |
| `POST` | `/internal/documents/{documentId}/analyze` | 문서 분석 Background Task 접수 |
| `POST` | `/internal/documents/{documentId}/reindex` | 기존 원문의 Chunk·Embedding 재생성 |
| `POST` | `/internal/document-evidence/search` | 허용 문서 범위의 의료 Evidence 검색 |
| `POST` | `/internal/ai-result-documents/generate` | Citation 기반 쉬운 설명·비교 답변 생성 |

</details>

<details>
<summary><b>💉 예방접종 일정 API</b></summary>

| Method | Endpoint | 설명 |
|---|---|---|
| `POST` | `/internal/vaccinations/schedule` | 생년월일·기준일·완료 차수를 바탕으로 접종 일정 계산 |

Spring은 `birthDate`, `asOf`, `completedDoses`를 보내고 AI Backend는 질병관리청 2026 일정 Data를 기준으로 `now`·`soon`·`past` 묶음, 다음 접종과 접종 개수를 반환합니다. 이 계산은 Database를 사용하지 않습니다.

- `VACCINATION_001`: 생년월일이 기준일보다 늦을 때 반환합니다.
- `VACCINATION_002`: 일정 Data에 없는 접종 차수가 포함되었을 때 반환합니다.
- 알 수 없는 차수는 정렬·중복 제거된 `unknownDoseIds`로 함께 반환합니다.

```json
{
  "code": "VACCINATION_002",
  "message": "알 수 없는 접종 차수가 포함되어 있습니다.",
  "timestamp": "2026-11-09T17:42:06.479040+00:00",
  "unknownDoseIds": ["DTAP-9", "RV-7"]
}
```

</details>

<details>
<summary><b>🎁 복지 AI API</b></summary>

| Method | Endpoint | 설명 |
|---|---|---|
| `GET` | `/api/v1/welfare/ready` | 복지 AI Module 상태 확인 |
| `GET` | `/api/v1/welfare/programs` | Spring에 동기화된 복지 Catalog 조회 |
| `POST` | `/api/v1/welfare/chunks` | 복지 Program을 검색 단위로 Chunking |
| `POST` | `/api/v1/welfare/embeddings/sync` | 복지 Chunk Embedding 동기화 |
| `POST` | `/api/v1/welfare/search` | 복지 정책 Chunk 검색 |
| `POST` | `/api/v1/welfare/explanations` | 추천 판정에 대한 보호자용 설명 생성 |
| `POST` | `/api/v1/welfare/recommendations/mode` | FIRST·SECOND·PROACTIVE 추천 Mode 결정 |

복지 Embedding 동기화는 pgvector 저장에 실패하면 Memory Content Hash Store로 Fallback합니다.

</details>

<details>
<summary><b>⚠️ 공통 오류 응답</b></summary>

```json
{
  "code": "GLOBAL_002",
  "message": "요청 값이 올바르지 않습니다.",
  "timestamp": "2026-11-17T00:00:00+00:00"
}
```

- 요청 검증 실패는 `400`과 공통 Code로 변환합니다.
- 내부 인증 실패는 `401`로 반환합니다.
- 예상하지 못한 예외는 상세 정보를 숨기고 `500`으로 반환합니다.

</details>

---

## 6. 🧱 Code Architecture

```mermaid
flowchart TD
    API["API · FastAPI Router"] --> APP["Application · Use Case"]
    APP --> DOMAIN["Domain · Rule · Model"]
    APP --> PORTS["Application Ports"]
    INFRA["Infrastructure Adapter"] -. implements .-> PORTS
    INFRA --> DB[("PostgreSQL · pgvector")]
    INFRA --> STORAGE[("Local Document Storage")]
    INFRA --> PROVIDER["VLM · LLM · Embedding Provider"]
    PIPE["Document Pipeline Step"] --> DOMAIN
    APP --> PIPE
```

<details>
<summary><b>📂 Project Structure</b></summary>

```text
Dodam_AI_backend
├── app
│   ├── api
│   │   ├── error_handlers.py           # 공통 JSON Error
│   │   └── internal/v1                 # Spring 내부 호출 Router
│   ├── application
│   │   ├── ports                       # Repository·Storage·Model Interface
│   │   ├── services                    # Retry 등 Application Service
│   │   ├── use_cases                   # 분석·색인·검색·답변 생성
│   │   └── welfare                     # 복지 설명·검색·Mode
│   ├── contracts                       # Pydantic Request·Response
│   ├── core/config.py                  # 환경 변수 Settings
│   ├── domain
│   │   ├── ai_result                   # 질문 분류·Safety Check
│   │   ├── document                    # 추출·Chunk·후보 Domain Rule
│   │   ├── vaccination                 # 예방접종 일정 Data·계산 Engine
│   │   └── welfare                     # 복지 Context·Policy Chunk
│   ├── infrastructure
│   │   ├── db                          # SQLAlchemy Model·Repository
│   │   ├── medical_rag                 # pgvector Evidence 검색
│   │   ├── providers                   # VLM·LLM·Embedding Adapter
│   │   ├── storage                     # Local·S3 Storage Adapter
│   │   └── welfare                     # Spring Client·복지 Embedding
│   ├── pipelines/documents             # Page 분리·저장·추출·Chunk Step
│   └── main.py                         # FastAPI App·Client Lifecycle
├── prompts/document_extraction         # Version별 VLM Prompt
├── scripts                             # 분석·추출·재색인 Tool
├── tests
│   ├── unit
│   └── integration
├── .env.example
├── pyproject.toml
└── README.md
```

</details>

---

## 7. 🛠️ 기술 스택

<details open>
<summary><b>⚙️ Application · Data</b></summary>

| 분류 | 기술 | 용도 |
|---|---|---|
| Runtime | Python 3.12+ | AI Backend Runtime |
| API | FastAPI · Uvicorn | Internal HTTP API |
| Validation | Pydantic v2 · pydantic-settings | 계약·환경 변수 검증 |
| ORM | SQLAlchemy 2 | 분석·Page·Chunk Repository |
| Database | PostgreSQL · psycopg | 공용 서비스 Data 저장 |
| Vector Search | pgvector | 의료·복지 Embedding 검색 |
| Object Storage | Local File · Amazon S3 · boto3 | 원본 문서 읽기와 Rendering Page 저장 |
| HTTP Client | httpx | Spring Backend 내부 호출 |

</details>

<details>
<summary><b>🤖 AI · Document</b></summary>

| 분류 | 기술 | 용도 |
|---|---|---|
| Model SDK | OpenAI Python SDK | OpenAI Compatible LLM·VLM 호출 |
| PDF | PyMuPDF | PDF Page Rendering |
| Image | Pillow | JPG·PNG 전처리와 크기 제한 |
| Tokenizer | tiktoken | Chunk Token 계산 |
| Embedding | OpenAI Embedding API | 768 Dimension Vector 생성 |
| RAG | pgvector Cosine Distance | 허용 문서 Chunk 검색 |

</details>

<details>
<summary><b>✅ Quality</b></summary>

| 분류 | 기술 |
|---|---|
| Test | Pytest |
| Lint | Ruff |
| Test Strategy | Unit · Integration · Manual Contract Server |

</details>

---

## 8. 🚀 Local 실행

<details open>
<summary><b>📦 설치</b></summary>

```bash
git clone https://github.com/ieum-connect/Dodam_AI_backend.git
cd Dodam_AI_backend

uv sync --extra dev
```

`uv`를 사용하지 않는 환경에서는 Python 3.12 Virtual Environment에 Project와 `dev` Extra를 설치합니다.

```bash
python -m pip install -e ".[dev]"
```

</details>

<details open>
<summary><b>▶️ Server 실행</b></summary>

```bash
cp .env.example .env
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Windows PowerShell에서는 다음과 같이 복사합니다.

```powershell
Copy-Item .env.example .env
uv run uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

Health Check:

```bash
curl http://127.0.0.1:8000/health
```

```json
{"status":"ok"}
```

</details>

<details>
<summary><b>🧪 API Key 없이 Contract 연동 확인</b></summary>

```bash
python -m tests.manual_contract_server
```

- VLM 응답만 고정된 Example로 대체합니다.
- 파일 읽기, Page 저장, 결과 검증과 후보 DB 저장은 실제 경로를 실행합니다.
- 실제 AI 판독 품질이나 Provider 호출을 검증하지 않습니다.
- 배포 Entry Point로 사용하지 않습니다.
- Frontend에는 `VITE_AI_TEST_MODE=true`를 설정하여 Example 응답임을 표시합니다.

</details>

---

## 9. 🔐 환경 변수

<details open>
<summary><b>🗄️ Database · Storage · 내부 통신</b></summary>

```env
DATABASE_URL=postgresql+psycopg://dodam:dodam_local@127.0.0.1:5433/dodam
STORAGE_LOCAL_ROOT=C:/path/to/Dodam_backend/data/object-storage
STORAGE_PROVIDER=local
STORAGE_S3_BUCKET=
STORAGE_S3_REGION=
AI_INTERNAL_TOKEN=change-me

SPRING_BACKEND_BASE_URL=http://localhost:8081
SPRING_BACKEND_TIMEOUT_SECONDS=10
```

Spring Backend와 다음 값이 일치해야 합니다.

- Database
- Local Mode에서는 `STORAGE_LOCAL_ROOT`
- S3 Mode에서는 Bucket 이름과 실제 Region
- `AI_INTERNAL_TOKEN`

Local 개발은 `STORAGE_PROVIDER=local`을 사용합니다. 비공개 S3를 사용하는 배포에서는 다음처럼 설정합니다.

```env
STORAGE_PROVIDER=s3
STORAGE_S3_BUCKET=dodam-private-documents
STORAGE_S3_REGION=ap-northeast-2
```

S3 Bucket과 Region은 Spring의 `DODAM_STORAGE_S3_BUCKET`, `DODAM_STORAGE_S3_REGION`과 같아야 합니다. S3 Mode에서 Bucket 또는 Region이 비어 있으면 잘못된 설정으로 기동하는 대신 Server 시작 단계에서 실패합니다.

</details>

<details>
<summary><b>🧠 LLM · VLM</b></summary>

```env
LLM_API_KEY=
LLM_PROVIDER=openai
LLM_BASE_URL=https://api.openai.com/v1
LLM_MODEL=
LLM_TIMEOUT_SECONDS=90
LLM_MAX_RETRIES=2
LLM_REASONING_EFFORT=low

OPENAI_API_KEY=
OPENAI_VLM_MODEL=
OPENAI_VLM_REASONING_EFFORT=high
OPENAI_VLM_MAX_OUTPUT_TOKENS=16000
OPENAI_VLM_TIMEOUT_SECONDS=180
OPENAI_VLM_CONNECT_TIMEOUT_SECONDS=10
OPENAI_VLM_MAX_RETRIES=2
OPENAI_VLM_RETRY_BACKOFF_SECONDS=1
```

- `OPENAI_API_KEY`는 문서 Page VLM 추출에 사용합니다.
- `LLM_API_KEY`가 비어 있으면 일반 LLM Client가 `OPENAI_API_KEY`를 대신 사용합니다.
- API Key가 없으면 실제 문서 판독을 실패로 기록하며 가짜 성공 결과를 반환하지 않습니다.

</details>

<details>
<summary><b>🔎 Embedding · RAG</b></summary>

```env
EMBEDDING_API_KEY=
EMBEDDING_MODEL=text-embedding-3-small
EMBEDDING_DIMENSIONS=768
EMBEDDING_TIMEOUT_SECONDS=30
EMBEDDING_MAX_RETRIES=2
EMBEDDING_RETRY_BACKOFF_SECONDS=1

MEDICAL_RAG_SEARCH_LIMIT=3
MEDICAL_RAG_MAX_EVIDENCE=20
```

`EMBEDDING_DIMENSIONS`는 Database의 `vector(768)` Column과 일치해야 하며 다른 값은 Server 기동 시 거부합니다.

</details>

<details>
<summary><b>📄 문서 처리 제한</b></summary>

```env
DOCUMENT_RENDER_DPI=150
DOCUMENT_MAX_PAGES=30
DOCUMENT_MAX_PAGE_SIDE_PX=2048
```

</details>

---

## 10. ♻️ 재색인과 운영 복구

<details open>
<summary><b>🔁 문서 하나 재색인</b></summary>

```bash
python -m scripts.reindex_documents <documentId>
```

또는 Spring 내부 호출을 통해 Background Task를 접수합니다.

```text
POST /internal/documents/{documentId}/reindex
```

</details>

<details>
<summary><b>🧹 누락된 모든 Embedding 복구</b></summary>

```bash
python -m scripts.reindex_documents --missing
```

- 최신 분석에 Chunk가 없거나 Embedding이 빠진 문서를 처리합니다.
- 저장된 Page 원문을 사용하므로 VLM을 다시 호출하지 않습니다.
- 문서와 분석 상태는 변경하지 않습니다.
- 문서마다 Embedding API 비용이 발생할 수 있습니다.
- `EMBEDDING_API_KEY`가 없으면 대상 조회 전에 종료합니다.

</details>

<details>
<summary><b>⚠️ 현재 운영상 주의점</b></summary>

- Background Worker가 비정상 종료되면 `RUNNING`·`PROCESSING` 상태가 남을 수 있습니다.
- 같은 분석 작업의 자동 재시작은 현재 지원하지 않습니다.
- 재분석 시 사라진 `PENDING` 후보는 취소하지만 이미 승인·거절한 보호자 결정은 유지합니다.
- 성공한 문서 분석 뒤 색인만 실패한 경우 문서 결과는 보존하고 재색인으로 복구합니다.

</details>

---

## 11. ✅ Test와 검증

<details open>
<summary><b>🧪 실행 명령어</b></summary>

```bash
uv run ruff check .
uv run pytest
```

또는 활성화된 Virtual Environment에서 실행합니다.

```bash
ruff check .
pytest
```

</details>

<details>
<summary><b>📋 주요 검증 범위</b></summary>

- PDF·Image Page 분리, 크기 제한과 손상·암호화 파일 처리
- VLM Strict Schema, Page 좌표와 주민등록번호 Masking
- Retry 대상 Error와 지수 Backoff
- 문서 상태 `READY`·`NEEDS_REVIEW`·`FAILED` 전이
- 일정·복약·주의 후보의 원문 Grounding과 안정적인 ID
- Chunk Token 상한, 겹침과 문장·행 경계 보존
- Embedding Batch, Retry, Model·Dimension 검증
- pgvector 기반 문서 검색과 Patient·Document Filter
- 의료 판단 질문 차단과 응급 안내
- Citation이 없는 생성 문장 제거
- 복지 설명의 점수 보존과 Hallucination Fallback
- Internal Token, 요청 검증과 공통 Error 응답
- Database Repository와 Storage Path 탈출 방지
- Local·S3 Storage의 원본 읽기와 Rendering Page 저장
- 예방접종 일정 묶음, 다음 접종과 알 수 없는 차수 Error 응답

</details>

---

## 12. 🤝 개발 컨벤션

<details>
<summary><b>📚 공통 컨벤션 문서</b></summary>

- [브랜치 컨벤션](https://github.com/ieum-connect/Dodam_backend/blob/develop/docs/convention/BRANCH_CONVENTION.md)
- [커밋 컨벤션](https://github.com/ieum-connect/Dodam_backend/blob/develop/docs/convention/COMMIT_CONVENTION.md)
- [PR 컨벤션](https://github.com/ieum-connect/Dodam_backend/blob/develop/docs/convention/PR_CONVENTION.md)

</details>

```mermaid
flowchart LR
    ISSUE["📌 Issue"] --> BRANCH["🌿 Branch"] --> DEVELOP["💻 Develop"]
    DEVELOP --> TEST["✅ Test"] --> COMMIT["📝 Commit"]
    COMMIT --> PR["🔍 Pull Request"] --> REVIEW["👀 Code Review"] --> MERGE["🎉 Merge"]
```

---

<div align="center">

### 문서의 근거를 지키고, 보호자가 이해할 수 있는 설명을 만드는 도담 AI 🤖💗

Made by **Team IEUM**

</div>
