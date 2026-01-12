import asyncio
from datetime import datetime

from typing import Any, Dict, Optional, Literal, List, Annotated

from tavily import AsyncTavilyClient

from langchain.tools import tool, InjectedToolArg
from langchain_core.runnables import RunnableConfig

from src.Legacy_Workflow.state import Section

def get_search_params(search_api: str, search_api_config: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """
    search_api_config을 필터링하여 지정된 검색 API에서 허용되는 파라미터만 포함하도록 하는 함수.

    Args:
        search_api (str): The search API identifier (e.g., "exa", "tavily").
        search_api_config (Optional[Dict[str, Any]]): 검색 API에 대한 파라미터 구성 

    Returns:
        Dict[str, Any]: A dictionary of parameters to pass to the search function.
    """
    # 각 검색 API에 대한 허용된 파라미터 정의
    SEARCH_API_PARAMS = {
        "tavily": ["max_results", "topic", "include_raw_content"],
        "arxiv": ["load_max_docs", "get_full_documents", "load_all_available_meta"],
        "pubmed": ["top_k_results", "email", "api_key", "doc_content_chars_max"],
        "googlesearch": ["max_results"],
    }

    # 검색 API에 허용된 파라미터 가져오기
    accepted_params = SEARCH_API_PARAMS.get(search_api, [])

    # search_api_config이 None인 경우 빈 딕셔너리 반환
    if not search_api_config:
        return {}

    # 허용된 파라미터만 포함하는 새 딕셔너리 생성
    return {k: v for k, v in search_api_config.items() if k in accepted_params}



async def tavily_search_async(search_queries, 
                              max_results: int = 1, 
                              topic: Literal["general", "news", "finance"] = "general", 
                              include_raw_content: bool = True):
    """
    Tavily API 비동기 검색 

    Args:
        search_queries (List[str]): 검색할 쿼리 리스트 
        max_results (int): 최대 검색 결과 수
        topic (Literal["general", "news", "finance"]): 필터링 주제
        include_raw_content (bool): 전체 페이지 콘텐츠 포함 여부

    Returns:
            List[dict]: Tavily API 검색 결과 리스트:
                {
                    'query': str,
                    'follow_up_questions': None,      
                    'answer': None,
                    'images': list,
                    'results': [                     # List of search results
                        {
                            'title': str,            # Title
                            'url': str,              # URL
                            'content': str,          # 요약 내용
                            'score': float,          # 관련도 점수
                            'raw_content': str|None  # 전체 페이지(마크다운)
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

    # 비동기적으로 모든 검색 작업 실행
    search_docs = await asyncio.gather(*search_tasks)
    return search_docs


TAVILY_SEARCH_DESCRIPTION = (
    "A search engine optimized for comprehensive, accurate, and trusted results. "
    "Useful for when you need to answer questions about current events."
)

@tool(description=TAVILY_SEARCH_DESCRIPTION)
async def tavily_search(
    queries: List[str],
    max_results: Annotated[int, InjectedToolArg] = 1,   # InjectedToolArg: 툴 호출 시 LLM에게 자동으로 값을 주입
    topic: Annotated[Literal["general", "news", "finance"], InjectedToolArg] = "general",
    include_raw_content: bool = True,
    config: RunnableConfig = None,
) -> str:
    
    """
    Tavily 검색 API에서 결과 출력 

    Args:
        queries (List[str]): 검색 쿼리 리스트
        max_results (int): 최대 검색 결과 수
        topic (Literal['general', 'news', 'finance']): 검색 주제

    Returns:
        str: 포맷팅된 검색 결과 문자열
    """
    
    search_results = await tavily_search_async(
        queries,
        max_results=max_results,
        topic=topic,
        include_raw_content=include_raw_content,
    )
    
    # 중복 URL 제거 
    unique_results = {}
    for response in search_results:
        for res in response['results']:
            url = res['url']
            
            if url not in unique_results:
                unique_results[url] = res


    # 웹 검색 결과 포맷팅
    formatted_output = ""
    for i, (url, result) in enumerate(unique_results.items()):
        formatted_output += f"\n\n===== SOURCE {i+1}: {result['title']} =====\n\n"
        formatted_output += f"URL: {url}\n\n"
        formatted_output += f"SUMMARY:\n{result['content']}\n\n"
        if result.get('raw_content'):
            formatted_output += f"FULL CONTENT:\n{result['raw_content'][:30000]}"  # Limit content size
        formatted_output += "\n\n" + "-" * 80 + "\n"
        
        
    if unique_results:
        return formatted_output
    
    else:
        return "No search results found."





async def run_search(search_api: str, query_list: list[str], web_param_filter: dict) -> str:
    """지정한 검색 API를 선택하고 실행.

    Args:
        search_api: 사용할 검색 API 이름
        query_list: 실행할 검색 쿼리 목록
        web_param_filter: 검색 API에 전달할 파라미터

    Returns:
        검색 결과를 포맷팅한 문자열

    Raises:
        ValueError: 지원하지 않는 검색 API가 지정된 경우
    """
    
    if search_api == "tavily":
        # Tavily search tool used with both workflow and agent 
        # and returns a formatted source string
        return await tavily_search.ainvoke({'queries': query_list, **web_param_filter})
    
    # elif search_api == "duckduckgo":
    #     # DuckDuckGo search tool used with both workflow and agent 
    #     return await duckduckgo_search.ainvoke({'search_queries': query_list})
    

    # elif search_api == "arxiv":
    #     search_results = await arxiv_search_async(query_list, **web_param_filter)
        
    # elif search_api == "pubmed":
    #     search_results = await pubmed_search_async(query_list, **web_param_filter)
        

    # else:
    #     raise ValueError(f"Unsupported search API: {search_api}")

    # return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000, deduplication_strategy="keep_first")

    

def get_today() -> str:
    """현재 날짜를 "2025-10-10" 형식의 문자열로 반환하는 함수."""
    return datetime.now().strftime("%Y-%m-%d")


def format_sections(sections: list[Section]) -> str:
    """Research 섹션들을 포맷팅"""
    formatted_str = ""
    for idx, section in enumerate(sections, 1):
        formatted_str += f"""
            {'='*60}
            Section {idx}: {section.name}
            {'='*60}
            Description:
            {section.description}
            Requires Research: 
            {section.research}

            Content:
            {section.content if section.content else '[Not yet written]'}

"""
    return formatted_str
