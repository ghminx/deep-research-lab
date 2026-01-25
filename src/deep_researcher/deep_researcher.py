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
    ConductResearch,
    ResearchComplete,
    
    AgentInputState,
    AgentState,
    SupervisorState
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
    get_today,
    think_tool,
    get_notes_from_tool_calls
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
    

# async def write_research_brief(state: AgentState, config: RunnableConfig) -> Command[Literal["research_supervisor"]]:
async def write_research_brief(state: AgentState, config: RunnableConfig):
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
    

async def supervisor(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor_tools"]]:
    """
    연구 전략을 계획하고 연구자들에게 작업을 위임하는 리드 연구 감독자

    Supervisor는 연구 브리프를 분석하고 연구를 관리 가능한 작업들로 분해하는 방법을 결정
    전략적 계획을 위해 think_tool을, 하위 연구자에게 작업을 위임하기 위해 ConductResearch를,
    결과에 만족할 때 ResearchComplete를 사용할 수 있음

    Args:
        state: 메시지와 연구 컨텍스트가 포함된 현재 supervisor 상태
        config: 모델 설정이 포함된 런타임 구성
        
    Returns:
        도구 실행을 위해 supervisor_tools로 진행하는 Command
    """
    configurable = Configuration.from_runnable_config(config)
    
    # 사용가능한 Tool 정의 
    lead_researcher_tools = [ConductResearch, ResearchComplete, think_tool]

    # 모델 설정
    research_model_name = configurable.research_model
    model_max_token = configurable.research_model_max_tokens
    research_model = (init_chat_model(model=research_model_name, max_tokens=model_max_token)
                    .bind_tools(lead_researcher_tools)
                    .with_retry(stop_after_attempt=configurable.max_structured_output_retries)
                    )
    
    # Supervisor 응답 생성(도구 호출)
    supervisor_messages = state.get("supervisor_messages", [])
    response = await research_model.ainvoke(supervisor_messages)
    
    return Command(
        goto="supervisor_tools",
        update={"supervisor_messages": [response],
                "research_iterations": state.get("research_iterations", 0) + 1})
    
    

async def supervisor_tools(state: SupervisorState, config: RunnableConfig) -> Command[Literal["supervisor", "__end__"]]:
    """Supervisor에서 호출한 도구를 실행

    3가지의 도구를 실행함 
    
    1. think_tool - 대화를 지속하는 전략적 사고
    2. ConductResearch - 하위 연구원에게 연구 작업을 위임
    3. ResearchComplete - 연구 단계 완료 신호

    각 도구 호출은 ToolMessage로 변환되어 supervisor에게 다시 전송되며,
    supervisor가 진행 상황을 추적하고 다음 단계를 계획할 수 있게함

    Args:
        state: 현재 supervisor 상태
        config: 런타임 구성 정보

    Returns:
        다음 supervisor 단계로 돌아가거나 연구 완료로 진행하는 Command
    """

    # 설정 및 현재 State 추출 
    configurable = Configuration.from_runnable_config(config)
    research_iterations = state.get("research_iterations", 0)
    supervisor_messages = state.get("supervisor_messages", [])
    recent_message = supervisor_messages[-1]
    
    # Research 단계 종료 기준 
    allowed_iterations = research_iterations > configurable.max_researcher_iterations # 최대 연구 반복 횟수 초과
    no_tool_calls = not recent_message.tool_calls # 도구 호출이 없음
    # research_complete = any(
    #     call.tool_name == "ResearchComplete" 
    #     for call in recent_message.tool_calls
    # )
    
    # Complete tool 호출 시 
    research_complete = False 
    
    for tool_call in recent_message.tool_calls:
        if tool_call['tool_name'] == "ResearchComplete":
            research_complete = True
            break    
        
    # 연구 완료 조건 충족 시 END 상태로 이동
    if allowed_iterations or no_tool_calls or research_complete:
        return Command(
            goto=END,
            update={
                "notes": get_notes_from_tool_calls(supervisor_messages),
                "research_brief": state.get("research_brief", "")
            }
        )


# 메인 Graph
deep_researcher_builder = StateGraph(
    AgentState, 
    input=AgentInputState, 
    config_schema=Configuration
)

# main workflow nodes 정의
deep_researcher_builder.add_node("clarify_with_user", clarify_with_user)           # User clarification phase
deep_researcher_builder.add_node("write_research_brief", write_research_brief)     # Research planning phase

# main workflow edges 정의
deep_researcher_builder.add_edge(START, "clarify_with_user")                       # Entry point

# main workflow Complie
deep_researcher = deep_researcher_builder.compile()

async def run():
    response = await deep_researcher.ainvoke({"messages": '딥러닝의 역사에 대해서 포괄적으로 조사해줘 추가질문은하지마'}, config=RunnableConfig())
    
    return response

if __name__ == "__main__":
    response = asyncio.run(run())


supervisor_response = asyncio.run(supervisor(response, RunnableConfig()))
