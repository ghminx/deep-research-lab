import operator

from typing import Annotated, Optional

from langchain.messages import MessageLikeRepresentation
from langgraph.graph import MessagesState
from pydantic import BaseModel, Field
from typing_extensions import TypedDict


# ====================
# Structured Outputs
# ====================
class ClarifyWithUser(BaseModel):
    """사용자의 연구 범위 명확화를 위한 구조화된 출력"""
    
    need_clarification: bool = Field(
        description="질문 명확화 여부",
    )
    question: str = Field(
        description="사용자에게 보고서 범위를 명확히 하기 위해 묻는 질문, 사용자에게 추가 질문을 함",
    )
    verification: str = Field(
        description="사용자가 필요한 정보를 제공한 후 연구를 시작할 것이라는 메시지를 확인, 명확화가 필요없을때",
    )
    
    
class ResearchQuestion(BaseModel):
    """연구 가이드를 세우기 위한 질문 및 브리프"""
    
    research_brief: str = Field(
        description="연구를 수행하는 데 기준이 될 질문",
    )

class ConductResearch(BaseModel):
    """특정 주제에 대한 연구를 수행 하는 도구"""
    research_topic: str = Field(
        description="The topic to research. Should be a single topic, and should be described in high detail (at least a paragraph).",
    )

class ResearchComplete(BaseModel):
    """Call this tool to indicate that the research is complete."""
    

# ====================
# State Definitions
# ====================
def override_reducer(current_value, new_value):
    """필요 조건에 따라 Override(덮어쓰기) 하거나 기존의 값에 누적하는 Reducer function"""
    
    if isinstance(new_value, dict) and new_value.get("type") == "override":
        return new_value.get("value", new_value)
    else:
        return operator.add(current_value, new_value)
    
class AgentInputState(MessagesState):
    """InputState is only 'messages'."""
    
class AgentState(MessagesState):
    """Main agent state containing messages and research data."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: Optional[str]                                                       # Optional[str] = str 이거나 None
    raw_notes: Annotated[list[str], override_reducer] = []                              # 원본 검색 결과 
    notes: Annotated[list[str], override_reducer] = []                                  # Researcher가 정리한 요약본
    final_report: str
    
class SupervisorState(TypedDict):
    """State for the supervisor that manages research tasks."""
    
    supervisor_messages: Annotated[list[MessageLikeRepresentation], override_reducer]
    research_brief: str
    notes: Annotated[list[str], override_reducer] = []
    research_iterations: int = 0
    raw_notes: Annotated[list[str], override_reducer] = []