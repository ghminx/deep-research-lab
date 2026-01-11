from typing import Annotated, List, TypedDict, Literal
from pydantic import BaseModel, Field
import operator


# LLM 출력 State 정의

class SearchQuery(BaseModel):
    search_query: str = Field(..., description="Query for web search.") 
    

class Queries(BaseModel):
    queries: List[SearchQuery] = Field(
        description="List of search queries.",
    )  
    
# # 이렇게 해도 문제 없을듯 
# class Queries(BaseModel):
#     queries: List[str] = Field(
#         description="List of search queries.",
#     )


class Section(BaseModel):
    name: str = Field(
        description="Name for this section of the report.",
    )
    description: str = Field(
        description="Brief overview of the main topics and concepts to be covered in this section.",
    )
    
    # 웹 검색 수행 여부
    research: bool = Field(
        description="Whether to perform web research for this section of the report."
    )
    content: str = Field(
        description="The content of the section."
    )   

class Sections(BaseModel):
    sections: List[Section] = Field(
        
        description="Sections of the report.",
    )




# Graph State 정의
class ReportStateInput(TypedDict):
    topic: str # 보고서 주제
    
class ReportStateOutput(TypedDict):
    final_report: str       # Final report
    source_str: str         # 웹 검색에서 포맷된 원본 콘텐츠 문자열

class ReportState(TypedDict):
    topic: str 
    feedback_on_report_plan: Annotated[list[str], operator.add]     # 보고서 계획에 대한 피드백 리스트
    sections: list[Section]                                         # 보고서 섹션 리스트
    completed_sections: Annotated[list, operator.add]               # 완료된 섹션들 (Send() API로 병렬 처리 후 합쳐짐)
    report_sections_from_research: str                              # 리서치로 작성된 섹션 내용 (최종 섹션 작성시 참고용)
    final_report: str 
    source_str: Annotated[str, operator.add]                        # 웹 검색 출처 문자열 configurable.include_source_str이 True일 때만 포함됨

class SectionState(TypedDict):
    topic: str 
    section: Section     
    search_iterations: int               # 검색 반복 횟수
    search_queries: list[SearchQuery]    # 검색 쿼리 리스트
    source_str: str                      # 포맷팅 된 웹 검색 결과 
    report_sections_from_research: str      # 완료된 섹션들 (최종 섹션 작성시 참고)
    completed_sections: list[Section]       # 완료된 섹션 (외부 그래프로 반환)
    
class SectionOutputState(TypedDict):
    completed_sections: list[Section] # Final key we duplicate in outer state for Send() API
    source_str: str # String of formatted source content from web search