
from typing import cast
from rich import print

from langchain_core.runnables import RunnableConfig
from langchain.chat_models import init_chat_model
from langchain.tools import tool, BaseTool
from langchain_core.runnables import RunnableConfig

from langgraph.types import Command, Send
from langgraph.graph import START, END, StateGraph


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
    FinishReport,
    FinishResearch
)

from src.Legacy_Multi_Agent.prompts import (
    SUPERVISOR_INSTRUCTIONS,
    
)




# Supervisor 도구 설정
async def get_supervisor_tools(config: RunnableConfig) -> list[BaseTool]:
    """Get supervisor tools based on configuration"""
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
    """Get researcher tools based on configuration"""
    
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

    return {
        "messages": [
            await llm_with_tools.ainvoke(
                [
                    {
                        "role": "system",
                        "content": system_prompt
                    }
                ]
                + messages
            )
        ]
    }