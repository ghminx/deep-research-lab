import os
from enum import Enum
from typing import Any, Optional

from langchain_core.runnables import RunnableConfig
from pydantic import BaseModel, Field


class SearchAPI(Enum):
    TAVILY = "tavily"
    ANTHROPIC = "anthropic"
    OPENAI = "openai"
    NONE = "none"
    
    
class Configuration(BaseModel):
    """Multi Agent 딥리서치 시스템 기본 설정"""
    
    # 기본 설정 
    search_api: SearchAPI = Field(default=SearchAPI.TAVILY, description="웹 검색에 사용할 검색 API")
    
    max_structured_output_retries: int = Field(default=2, description="구조화된 출력 최대 재시도 횟수")
    allow_clarification: bool = Field(default=True, description="연구를 시작하기 전에 사용자에게 명확한 질문을 할 수 있도록 허용할지 여부")
    max_concurrent_research_units: int = Field(default=3, description="동시에 실행할 수 있는 최대 연구 단위 수. 이렇게 하면 연구자는 여러 하위 에이전트를 사용하여 연구를 수행할 수 있음, 동시성이 높아지면 속도 제한에 부딪힐 수 있음.")
    
    max_researcher_iterations: int = Field(default=3, description="Supervisor의 최대 연구 반복 횟수. Supervisor가 연구에 대해 반성하고 후속 질문을 하는 횟수")
    max_react_tool_calls: int = Field(default=5, description="Researcher가 각 섹션에 대해 수행할 수 있는 최대 도구 호출 횟수")
    
    # Summarization 모델 설정
    summarization_model: str = Field(default="openai:gpt-5-mini", description="웹 검색 결과 요약 모델")
    summarization_model_max_tokens: int = Field(default=8192, description="요약 모델의 최대 토큰 수")
    max_content_length: int = Field(default=50000, description="요약 전 웹페이지 콘텐츠의 최대 문자 길이")
    
    # Researcher 모델 설정
    research_model: str = Field(default="openai:gpt-5.1", description="Researcher 모델")
    research_model_max_tokens: int = Field(default=8192, description="Researcher 모델의 최대 토큰 수")
    
    # compression 모델 설정
    compression_model: str = Field(default="openai:gpt-5-mini", description="하위 에이전트의 연구 결과를 압축하는 모델, 압축 모델이 선택한 검색 API를 지원하는지 확인")
    compression_model_max_tokens: int = Field(default=8192, description="압축 모델의 최대 토큰 수")
    
    # Final Report 모델 설정
    final_report_model: str = Field(default="openai:gpt-5.1", description="최종 보고서 작성 모델")
    final_report_model_max_tokens: int = Field(default=8192, description="최종 보고서 작성 모델의 최대 토큰 수")
    
    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> "Configuration":
        """Create a Configuration instance from a RunnableConfig."""
        configurable = config.get("configurable", {}) if config else {}
        field_names = list(cls.model_fields.keys())
        values: dict[str, Any] = {
            field_name: os.environ.get(field_name.upper(), configurable.get(field_name))
            for field_name in field_names
        }
        return cls(**{k: v for k, v in values.items() if v is not None})