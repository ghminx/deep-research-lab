
import asyncio

from typing import cast, Literal, List
from rich import print

from langchain.chat_models import init_chat_model
from langchain.tools import tool, BaseTool
from langchain_core.runnables import RunnableConfig
from langchain.messages import AIMessage, HumanMessage

from langgraph.types import Command, Send
from langgraph.graph import START, END, StateGraph, MessagesState


from src.Legacy_Multi_Agent.config import Configuration

from src.Legacy_Multi_Agent.utils import (
    tavily_search,
    get_search_tool,
    get_today
)

from src.Legacy_Multi_Agent.state import (
    Section,
    Sections,
    Introduction,
    Conclusion,
    Question,
    FinishResearch,
    FinishReport,
    ReportState,
    ReportStateOutput,
    SectionState,
    SectionOutputState,
)

from src.Legacy_Multi_Agent.prompts import (
    SUPERVISOR_INSTRUCTIONS,
    
)




# Supervisor 도구 설정
async def get_supervisor_tools(config: RunnableConfig) -> list[BaseTool]:
    """tool을 설정하고 리스트를 반환"""
    configurable = Configuration.from_runnable_config(config)
    
    search_tool = get_search_tool(config)
    
    tools = [tool(Sections), tool(Introduction), tool(Conclusion), tool(FinishReport)]
    
    if configurable.ask_for_clarification:
        tools.append(tool(Question))
    
    if search_tool is not None:
        tools.append(search_tool)  # Add search tool, if available
    
    return tools

# Researcher 도구 설정
async def get_research_tools(config: RunnableConfig) -> List(BaseTool):
    """tool을 설정하고 리스트를 반환"""
    
    search_tool = get_search_tool(config)
    
    
    tools = [tool(Section), tool(FinishResearch)]
    
    if search_tool is not None:
        tools.append(search_tool)  
    
    return tools

async def supervisor(state: ReportState, config: RunnableConfig):
    """
    Supervisor LLM이 다음 행동을 결정
    
    역할:
    - 보고서 작성 전체 흐름을 관리하는 리더 역할
    - LLM이 도구를 선택하여 다음 행동 결정 (Sections, Introduction, Conclusion, FinishReport 등)
    
    흐름:
    1. 현재 messages 가져오기
    2. 리서치 완료 시 intro/conclusion 작성 신호 추가
    3. 도구 목록 바인딩 (bind_tools)
    4. LLM 호출 → tool_calls 반환
    
    Args:
        state: 현재 상태 (messages, completed_sections, final_report 등)
        config: 모델 설정 및 API 키
        
    Returns:
        {"messages": [AI 응답 (tool_calls 포함)]}
    """
    
    # State 가져오기
    messages = state["messages"]
    
    # 설정 가져오기
    configurable = Configuration.from_runnable_config(config)
    supervisor_model =configurable.supervisor_model
    
    supervisor_llm = init_chat_model(supervisor_model)

    # 리서치가 완료되었으면 intro/conclusion 작성 신호 추가
    if state["completed_sections"] and not state['final_report']:
        research_complete_message = {
            "role": "user", 
            "content": "Research is complete. Now write the introduction and conclusion for the report. Here are the completed main body sections: \n\n" + "\n\n".join([s.content for s in state["completed_sections"]])}
        
        # 기존 메세지에 완료 메세지 추가
        messages = messages + [research_complete_message]
        
    # 도구 목록 설명 
    supervisor_tool_list = await get_supervisor_tools(config)
    
    llm_with_tools = supervisor_llm.bind_tools(
        supervisor_tool_list,
        parallel_tool_calls=True,
        tool_choice="any",  # 최소 하나의 도구 실행 
        )
    
    # Supervisor LLM 프롬프트 
    system_prompt = SUPERVISOR_INSTRUCTIONS.format(today=get_today())

    response = await llm_with_tools.ainvoke(([{"role": "system", "content": system_prompt}]) + messages)
    
    return {"messages": [response]}    
    

    
async def supervisor_tools(state: ReportState, config: RunnableConfig) -> Command[Literal["supervisor", "research_team", END]]:
    """
        Supervisor가 선택한 도구를 실행하고 결과에 따라 다음 노드로 라우팅

        도구별 동작:
        - Sections: 각 섹션을 research_team으로 병렬 전송
        - Question: 사용자에게 질문 후 종료
        - FinishReport: 보고서 완료, 종료
        - Introduction: 서론 저장 후 supervisor로 돌아감
        - Conclusion: 최종 보고서 조립 후 supervisor로 돌아감
        - 검색 도구: 검색 수행 후 supervisor로 돌아감

        Returns:
            Command: 다음 노드 지정 (supervisor, research_team, END)
    """
    
    configurable  = Configuration.from_runnable_config(config)
    
    # tool list 가져오기 
    supervisor_tool_list = await get_supervisor_tools(config)
    
    # tool name으로 도구 매핑 {Section: StructuredTool()}
    supervisor_tools_by_name = {tool.name: tool for tool in supervisor_tool_list}
    
    # search tool 이름 추출 -> {'tavily_search'}
    search_tool_names = set()
    for tool in supervisor_tool_list:
        if tool.metadata is not None and tool.metadata['type'] == 'search':
            search_tool_names.add(tool.name)
    
    # state["messages"][-1] : AIMessage의 tool_calls
    # for tool_call in state["messages"][-1].tool_calls:


config = {"configurable": {
                           "search_api": "tavily",
                           'ask_for_clarification': False,
                           }}

supervisor_response = {
    'messages': [
        HumanMessage(
            content='MCP에 대해서 알려줘',
            additional_kwargs={},
            response_metadata={},
            id='3d8da9f9-2431-4ab6-a6d2-7a6f86309f45'
        ),
        AIMessage(
            content='',
            additional_kwargs={'refusal': None},    
            response_metadata={
                'token_usage': {
                    'completion_tokens': 544,       
                    'prompt_tokens': 1042,
                    'total_tokens': 1586,
                    'completion_tokens_details': {  
                        'accepted_prediction_tokens': 0,
                        'audio_tokens': 0,
                        'reasoning_tokens': 448,    
                        'rejected_prediction_tokens': 0
                    },
                    'prompt_tokens_details': {      
                        'audio_tokens': 0,
                        'cached_tokens': 1024       
                    }
                },
                'model_provider': 'openai',
                'model_name': 'gpt-5-2025-08-07',   
                'system_fingerprint': None,
                'id':
                'chatcmpl-CypTvnZeAoXZxWP24yNk22EoGLEnZ',
                'service_tier': 'default',
                'finish_reason': 'tool_calls',      
                'logprobs': None
            },
            id='lc_run--019bc996-02f9-7893-8985-23ea34871988-0',
            tool_calls=[
                {
                    'name': 'tavily_search',        
                    'args': {
                        'queries': [
                            'MCP meaning overview', 
                            'Model Context Protocol MCP overview 2024 2025',
                            'Anthropic MCP protocol tools servers clients',
                            'OpenAI MCP support',   
                            'Microsoft Certified    Professional MCP overview',
                            'MCP manufacturing      chemical polyols MCP acronym',
                            'MCP in gaming Master   Control Program Tron',
                            'MCP Korean explanation 모델 컨텍스트 프로토콜'
                        ],
                        'include_raw_content': True 
                    },
                    'id':
'call_Vg5dbTQneRKCupVzCaYQImpA',
                    'type': 'tool_call'
                }
            ],
            invalid_tool_calls=[],
            usage_metadata={
                'input_tokens': 1042,
                'output_tokens': 544,
                'total_tokens': 1586,
                'input_token_details': {
                    'audio': 0,
                    'cache_read': 1024
                },
                'output_token_details': {
                    'audio': 0,
                    'reasoning': 448
                }
            }
        )
    ]
}



for tool_call in supervisor_response["messages"][-1].tool_calls:
    print(tool_call['name'])
