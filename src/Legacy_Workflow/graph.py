
from rich import print 
from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain.messages import SystemMessage, HumanMessage
from langchain_core.messages import SystemMessage, HumanMessage


from src.Legacy_Workflow.state import (
    ReportStateInput,
    
    Queries,
    
    
)

from src.Legacy_Workflow.prompts import (
    report_planner_query_writer_instructions,
)


from src.Legacy_Workflow.utils import (
    get_search_params,
    get_today,
)

from src.Legacy_Workflow.config import Configuration



async def generate_report_plan(state: ReportStateInput, config: RunnableConfig):
    """섹션들로 구성된 보고서 계획을 생성하는 노드 
    
    1. 보고서 구조와 파라미터 설정을 가져옴 
    2. 계획을 수립하기 위해 검색 쿼리를 생성 
    3. 생성한 검색쿼리들로 웹 검색 수행 
    4. LLM을 통해 섹션(서론, 본론, 결론)들로 구성된 구조화된 계획을 생성 

    Args:
        state (ReportStateInput): 그래프 상태(Topic) 
        config (RunnableConfig): 모델, 검색 API등의 설정 정보 
        
    Returns:
        dict: 섹션들로 구성된 보고서 계획
    """
    
    # 입력값 - 주제 
    topic = state["topic"]
    
    
    ### human_feedback 노드에서 사용자가 계획을 거부하고 피드백을 줬을 때 사용 
    
    # 보고서 계획에 대한 피드백 리스트 가져오기
    feedback_list = state.get("feedback_on_report_plan", [])

    # 피드백을 하나의 문자열로 합치기
    feedback = " /// ".join(feedback_list) if feedback_list else ""
    
    
    # 설정값 가져오기 
    configurable = Configuration.from_runnable_config(config)
    report_structure = configurable.report_structure
    number_of_queries = configurable.number_of_queries
    search_api = configurable.search_api 
    search_api_config = configurable.search_api_config or {}   # 빈값이면 {} 으로 설정
    web_param_filter = get_search_params(search_api, search_api_config)
    

    # writer model 설정 
    writer_provider = configurable.writer_provider
    writer_model_name = configurable.writer_model
    writer_model_kwargs = configurable.writer_model_kwargs or {}
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    structured_llm = writer_model.with_structured_output(Queries)  # 구조화된 출력 형태 지정
    

    # 검색 쿼리 생성 프롬프트 포맷팅
    search_query_prompt = report_planner_query_writer_instructions.format(
        topic=topic,
        report_organization=report_structure,
        number_of_queries=number_of_queries,
        today = get_today()
    )

    # 검색 쿼리 생성
    results = await structured_llm.ainvoke([SystemMessage(content=search_query_prompt),
                                     HumanMessage(content="Generate search queries that will help with planning the sections of the report.")])
    
    query_list = [query.search_query for query in result.queries]


    # 웹 검색 실행 

from typing import Literal, List
from tavily import AsyncTavilyClient
import asyncio

async def tavily_search_async(search_queries, max_results: int = 1, topic: Literal["general", "news", "finance"] = "general", include_raw_content: bool = True):
    """
    Performs concurrent web searches with the Tavily API

    Args:
        search_queries (List[str]): List of search queries to process
        max_results (int): Maximum number of results to return
        topic (Literal["general", "news", "finance"]): Topic to filter results by
        include_raw_content (bool): Whether to include raw content in the results

    Returns:
            List[dict]: List of search responses from Tavily API:
                {
                    'query': str,
                    'follow_up_questions': None,      
                    'answer': None,
                    'images': list,
                    'results': [                     # List of search results
                        {
                            'title': str,            # Title of the webpage
                            'url': str,              # URL of the result
                            'content': str,          # Summary/snippet of content
                            'score': float,          # Relevance score
                            'raw_content': str|None  # Full page content if available
                        },
                        ...
                    ]
                }
    """
    tavily_async_client = AsyncTavilyClient()
    search_tasks = []
    for query in search_queries:
            search_tasks.append(
                tavily_async_client.search(
                    query,
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                    topic=topic
                )
            )

    # Execute all searches concurrently
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs


queries = ["에스티이노베이션 회사에 대해서"]
result = asyncio.run(tavily_search_async(queries))


from rich import print

formatted_output = f"Search results: \n\n"

# Deduplicate results by URL
unique_results = {}
for response in result:
    for result in response['results']:
        
        print(result)
        
