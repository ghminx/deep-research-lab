import os 
from enum import Enum
from dataclasses import dataclass, fields               # fields 는 dataclass의 필드 정보를 가져오는 함수
from typing import Any, Optional, Dict, Literal

from langchain_core.runnables import RunnableConfig

DEFAULT_REPORT_STRUCTURE = """Use this structure to create a report on the user-provided topic:

1. Introduction (no research needed)
   - Brief overview of the topic area

2. Main Body Sections:
   - Each section should focus on a sub-topic of the user-provided topic
   
3. Conclusion
   - Aim for 1 structural element (either a list or table) that distills the main body sections 
   - Provide a concise summary of the report"""


DEFAULT_REPORT_STRUCTURE = """이 구조를 사용하여 사용자가 제공한 주제에 대한 보고서를 작성하세요:

1. 서론 (연구 필요 없음)
   - 주제 영역에 대한 간략한 개요

2. 본문 섹션들:
   - 각 섹션은 사용자가 제공한 주제의 하위 주제에 초점을 맞춰야 합니다
   
3. 결론
   - 본문 섹션들을 정리하는 1개의 구조적 요소(목록 또는 표)를 목표로 합니다
   - 보고서의 간결한 요약을 제공합니다"""




class SearchAPI(Enum):
    TAVILY = 'tavily'
    ARXIV = 'arxiv'     # 논문 검색
    PUBMED = "pubmed"   # 의학 논문 검색
    DUCKDUCKGO = 'duckduckgo'
    GOOGLE_SEARCH = 'google_search'
    NONE = 'none'
    
    
   
@dataclass(kw_only=True)   # kw_only=True : 키워드 인자로만 생성 가능
class Configuration:
    # """워크플로우/그래프 기반 딥리서치 시스템 기본 설정"""
    
    # 기본 설정
    report_structure: str = DEFAULT_REPORT_STRUCTURE
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None         # Optional: 값이 있을수도 있고 None 일수도 있음, Dict[str, any] : key는 문자열, value는 아무거나                          # Optional: 값이 있을수도 없을수도 있음, Dict[str, Any]: 키는 문자열, 값은 아무거나
    process_search_results: Literal["summarize", "split_and_rerank"] | None = None      # Literal: 고정된 값들 중 하나만 가질 수 있음 
    
    # 요약 모델 설정 
    summarization_model_provider: str = "openai"
    summarization_model: str = "gpt-4o-mini"
    max_structured_output_retries: int = 3    # 구조화된 출력 재시도 최대 횟수  
    include_source_str: bool = False          # 검색한 원본 텍스트 포함 여부
    
    # Workflow-specific configuration
    number_of_queries: int = 2                # 생성할 검색 쿼리 수
    max_search_depth: int = 2                 # 검색 깊이 제한(잘못 검색 되었을 때 다시 검색하는 횟수)
    
    # 플래너 모델 설정 
    planner_provider: str = "openai"
    planner_model: str = "gpt-5-mini"
    planner_model_kwargs: Optional[Dict[str, Any]] = None
    
    # 보고서 작성 모델 설정
    writer_provider: str = "openai"
    writer_model: str = "gpt-5-mini"
    writer_model_kwargs: Optional[Dict[str, Any]] = None
    
    @classmethod
    def from_runnable_config(cls, config: Optional[RunnableConfig]) -> "Configuration":
    
        """Create a Configuration instance from a RunnableConfig."""
        configurable = (
            config["configurable"] if config and "configurable" in config else {}    # ()를 쓰면 줄바꿈 가능
        )
        values: dict[str, Any] = {
            f.name: os.environ.get(f.name.upper(), configurable.get(f.name))
            for f in fields(cls)
            if f.init
        }
        return cls(**{k: v for k, v in values.items() if v})  # **: 딕셔너리를 언패킹하여 키워드 인자로 전달  ex) {a: 1, b: 2} -> a=1, b=2 
    

