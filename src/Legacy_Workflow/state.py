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






# Graph State 정의
class ReportStateInput(TypedDict):
    topic: str # 보고서 주제
    

