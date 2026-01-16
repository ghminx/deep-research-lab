from typing import Annotated, List, TypedDict, Literal
from pydantic import BaseModel, Field
import operator

from langgraph.graph import MessagesState

# LLM 출력 State 정의
class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Research scope for this section of the report.",  # 보고서의 연구 범위
    )
    content: str = Field(
        description="The content of the section."
    )   
    
class Sections(BaseModel):
    """List of sections titles of the report."""
    sections: List[str] = Field(description="Sections of the report.",)
    
    
class Introduction(BaseModel):
    """Introduction to the report."""
    name: str = Field(
        description="Name for the report.",
    )
    content: str = Field(
        description="The content of the introduction, giving an overview of the report."
    )

class Conclusion(BaseModel):
    """Conclusion to the report."""
    name: str = Field(
        description="Name for the conclusion of the report.",
    )
    content: str = Field(
        description="The content of the conclusion, summarizing the report."
    )

class Question(BaseModel):
    """Ask a follow-up question to clarify the report scope."""
    question: str = Field(
        description="A specific question to ask the user to clarify the scope, focus, or requirements of the report."
    )
    
# No-op tool to indicate that the research is complete
class FinishResearch(BaseModel):
    """Finish the research."""

# No-op tool to indicate that the report writing is complete
class FinishReport(BaseModel):
    """Finish the report."""
    
# Graph State 정의
class ReportState(MessagesState):
    sections: List[str]
    completed_sections: Annotated[list[Section], operator.add]
    final_report: str
    source_str: str

class ReportStateOutput(MessagesState):
    final_report: str 
    source_str: str
    
class SectionState(MessagesState):
    section: str 
    completed_sections: list[Section] 
    source_str: str 

class SectionOutputState(TypedDict):
    completed_sections: list[Section] 
    source_str: str 
    
