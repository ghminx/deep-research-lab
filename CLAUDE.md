# CLAUDE.md

이 파일은 Claude Code (claude.ai/code)가 이 저장소의 코드를 다룰 때 참고하는 가이드입니다.

## 프로젝트 개요

Deep Research Lab은 자동화된 심층 연구 시스템을 연구하고 구현하기 위한 실험적 Python 프로젝트입니다. LangChain의 Open Deep Research 프로젝트에서 영감을 받아, 포괄적인 연구 보고서를 생성하는 다양한 아키텍처 접근 방식을 탐구합니다.

## 개발 명령어

```bash
# 의존성 설치 (uv 패키지 매니저 사용)
uv sync

# 메인 엔트리 포인트 실행
python main.py

# 가상환경 활성화 (Windows)
.venv\Scripts\activate
```

## 아키텍처

### 핵심 개념

이 프로젝트는 **계획 → 검색 → 종합 → 생성** 패턴을 따르는 심층 연구 워크플로우를 구현합니다:

1. **보고서 계획**: 주제를 기반으로 구조화된 검색 쿼리 생성
2. **웹 검색**: Tavily 또는 다른 검색 API를 통한 동시 검색 실행
3. **종합**: 검색 결과 처리 및 중복 제거
4. **보고서 생성**: 인용이 포함된 구조화된 마크다운 보고서 작성

### 주요 구현체

**Legacy Workflow** ([src/Legacy_Workflow/](src/Legacy_Workflow/)):
- LangGraph 패턴을 사용한 그래프 기반 워크플로우
- 반영(reflection) 루프가 포함된 순차적 섹션 생성
- Human-in-the-loop 친화적 설계
- 파일 구성:
  - `graph.py` - 메인 워크플로우 노드 및 그래프 구성
  - `state.py` - 그래프 상태용 Pydantic 모델 (ReportStateInput, Queries, SearchQuery)
  - `config.py` - 모델/API 설정이 포함된 Configuration 데이터클래스
  - `prompts.py` - LLM 프롬프트 템플릿
  - `utils.py` - 검색 파라미터 필터링 및 실행 헬퍼 함수

### 설정 시스템

[config.py](src/Legacy_Workflow/config.py)의 `Configuration` 데이터클래스 지원 항목:
- 다중 검색 API: Tavily, ArXiv, PubMed, DuckDuckGo, Google Search
- 플래너, 작성자, 요약 모델별 개별 설정
- 환경 변수 오버라이드 (필드명 대문자 사용)
- LangChain의 `RunnableConfig`를 통한 런타임 설정

### 검색 통합

이 프로젝트는 동시 쿼리를 통한 비동기 검색 실행을 사용합니다:
- 주요 검색 API: Tavily (`AsyncTavilyClient`)
- 결과는 URL 기준으로 중복 제거 후 LLM 소비용으로 포맷팅
- 검색 파라미터는 `get_search_params()`에서 API별로 필터링

## 기술 스택

- Python 3.13+
- LangChain / LangChain-OpenAI (LLM 오케스트레이션)
- Tavily (웹 검색)
- Rich (콘솔 출력)
- Pydantic (데이터 유효성 검사)
