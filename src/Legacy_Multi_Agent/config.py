import os
from enum import Enum
from dataclasses import dataclass, fields
from typing import Any, Optional, Dict, Literal

from langchain_core.runnables import RunnableConfig



class SearchAPI(Enum):
    TAVILY = "tavily"
    PERPLEXITY = "perplexity"
    DUCKDUCKGO = "duckduckgo"
    NONE = "none"
    
@dataclass(kw_only=True)
class Configuration:
    """Multi Agent 딥리서치 시스템 기본 설정"""
    
    # 기본 설정 
    search_api: SearchAPI = SearchAPI.TAVILY
    search_api_config: Optional[Dict[str, Any]] = None
    
    # 요약 모델 설정 
    summarization_model: str = "openai:gpt-5-mini"
    include_source_str: bool = False         
    
    # Multi Agent 설정 
    number_of_queries: int = 2
    supervisor_model: str = "openai:gpt-5-mini"
    researcher_model: str = "openai:gpt-5-mini"
    ask_for_clarification: bool = False # 사용자에게 명확한 질문 요청 여부
    
    
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
    

