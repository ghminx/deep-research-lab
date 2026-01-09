from datetime import datetime

from typing import Any, Dict, Optional



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
        "tavily": ["max_results", "topic"],
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
    
    elif search_api == "duckduckgo":
        # DuckDuckGo search tool used with both workflow and agent 
        return await duckduckgo_search.ainvoke({'search_queries': query_list})
    

    elif search_api == "arxiv":
        search_results = await arxiv_search_async(query_list, **web_param_filter)
        
    elif search_api == "pubmed":
        search_results = await pubmed_search_async(query_list, **web_param_filter)
        

    else:
        raise ValueError(f"Unsupported search API: {search_api}")

    return deduplicate_and_format_sources(search_results, max_tokens_per_source=4000, deduplication_strategy="keep_first")

    

def get_today() -> str:
    """현재 날짜를 "2025-10-10" 형식의 문자열로 반환하는 함수."""
    return datetime.now().strftime("%Y-%m-%d")