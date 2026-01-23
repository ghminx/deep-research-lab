import asyncio
from rich import print
from typing import Literal

from langchain.chat_models import init_chat_model

from langchain_core.messages import get_buffer_string

from langchain.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from langchain_core.runnables import RunnableConfig
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command

from src.deep_researcher.config import Configuration

from src.deep_researcher.state import (
    ClarifyWithUser,
    ResearchQuestion,
    
    AgentInputState,
    AgentState,
)

from src.deep_researcher.prompts import (
    clarify_with_user_instructions,
    compress_research_simple_human_message,
    compress_research_system_prompt,
    final_report_generation_prompt,
    lead_researcher_prompt,
    research_system_prompt,
    transform_messages_into_research_topic_prompt,    
)

from src.deep_researcher.utils import (
    get_today
)


async def clarify_with_user(state: AgentState, config: RunnableConfig) -> Command[Literal["write_research_brief", "__end__"]]:
    """사용자 메세지를 분석하여 연구범위가 불명확한 경우 추가 질문을 생성
    
    연구를 진행하기 전에 사용자의 요청에 대해 추가적인 명확화가 필요한지를 판단
    명확화가 비활성화되어 있거나 필요하지 않은 경우에는 바로 연구 단계로 진행
    
    Args:
        state: 사용자 메시지를 포함한 현재 에이전트 상태
        config: 모델 설정 및 선호도를 포함한 런타임 구성 정보
        
    Returns:
        Command to either end with a clarifying question or proceed to research brief
    """
    
    messages = state["messages"]
    configurable = Configuration.from_runnable_config(config)
    
    
    # 명확화를 건너뛰고 바로 연구 단계로 진행
    if not configurable.allow_clarification:
        return Command(goto="write_research_brief")
    
    # 모델 설정 
    clarification_model_name = configurable.research_model
    model_max_token = configurable.research_model_max_tokens
    
    # 구조화된 출력, 재시도 횟수 설정
    clarification_model = (
        init_chat_model(model=clarification_model_name, 
                        max_tokens=model_max_token)
        .with_structured_output(ClarifyWithUser)
        .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
    )

    # 명확화 메시지 생성
    """
    get_buffer_string: 여러개의 chat message(AIMessage, HumanMessage 들을 하나의 문자열로 변환)
    
    messages = [
    HumanMessage(content="이 보고서 만들어줘"),
    AIMessage(content="알겠습니다. 어떤 범위인가요?")]
    
    -> 
    Human: 이 보고서 만들어줘
    AI: 알겠습니다. 어떤 범위인가요?"""
    
    clarification_prompt = clarify_with_user_instructions.format(
        messages = get_buffer_string(messages),
        date = get_today())

    # LLM 호출 
    response = await clarification_model.ainvoke([HumanMessage(content=clarification_prompt)])
    
    # 명확화 필요 여부에 따른 분기
    if response.need_clarification:
        # END 상태로 이동하여 사용자에게 명확화 질문 전송
        return Command(
            goto=END,
            update={"messages": [AIMessage(content=response.question)]})
        
    else:
        # 검증 메세지와 함께 연구 범위 작성 단계로 이동
        return Command(
            goto="write_research_brief", 
            update={"messages": [AIMessage(content=response.verification)]})
    

async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
    """사용자 메시지를 분석하여 구조화된 연구 브리핑을 생성하고, Supervisor를 초기화하는 노드

    사용자의 메시지를 분석하여 Research Supervisor를 안내할 명확하고 집중된 연구 브리프(계획서)를 생성합니다.
    또한 적절한 프롬프트와 지침을 포함한 초기 연구 감독자 컨텍스트를 설정

    Args:
    - state: 사용자 메시지를 포함한 현재 에이전트 상태
    - config: 모델 설정을 포함한 런타임 구성 정보

    Returns:
    - 초기화된 컨텍스트와 함께 연구 감독자 단계로 진행하기 위한 명령"""
    
    messages = state.get("messages", [])
    configurable = Configuration.from_runnable_config(config)
    
    # 모델 설정 
    research_model_name = configurable.research_model
    model_max_token = configurable.research_model_max_tokens
    
    research_model = (init_chat_model(model=research_model_name, max_tokens=model_max_token)
                      .with_structured_output(ResearchQuestion)
                      .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
                      )
    
    # 연구 브리프 생성 프롬프트
    brief_prompt = transform_messages_into_research_topic_prompt.format(
        messages = get_buffer_string(messages),
        date = get_today())
    
    # LLM 호출 
    response = await research_model.ainvoke([HumanMessage(content=brief_prompt)])


    # Research Berief와 Instructions을 Research Supervisor에 전달
    supervisor_system_prompt = lead_researcher_prompt.format(
        date=get_today(),
        max_concurrent_research_units=configurable.max_concurrent_research_units,
        max_researcher_iterations=configurable.max_researcher_iterations
    )
    
    return Command(
        goto="research_supervisor", 
        update={
            "research_brief": response.research_brief,
            "supervisor_messages": {
                "type": "override",
                "value": [
                    SystemMessage(content=supervisor_system_prompt),
                    HumanMessage(content=response.research_brief)
                ]
            }
        }
    )
    
      
# state = {"messages": [HumanMessage(content="딥러닝의 역사")]}
# response = asyncio.run(clarify_with_user(state, RunnableConfig()))

# response

