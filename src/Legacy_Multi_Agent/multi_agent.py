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
    RESEARCH_INSTRUCTIONS
    
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
async def get_research_tools(config: RunnableConfig) -> list[BaseTool]:
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
    system_prompt = SUPERVISOR_INSTRUCTIONS.format(number_of_queries = configurable.number_of_queries,
                                                   today=get_today())

    # LLM이 어떤 도구를 사용할지 결정
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
    
    result = []                      # ToolMessage들을 담을 리스트 (LLM에게 tool 실행 결과 전달용)
    sections_list = []               # Sections tool이 호출되면 섹션 목록 저장
    intro_content = None             # Introduction tool 실행 결과
    conclusion_content = None        # Conclusion tool 실행 결과
    source_str = ""                  # 검색 결과 문자열 (평가용)
                        
    
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
            
    
    # 가장 최신 메세지에 있는 tool_calls를 가져와 도구를 하나씩 실행(LLM을 동작하는게 아님 도구를 실행) 
    for tool_call in state["messages"][-1].tool_calls:
        
        tool = supervisor_tools_by_name[tool_call["name"]]
        
        try:
            observation = await tool.ainvoke(tool_call["args"], config)
        except NotImplementedError:
            observation = tool.invoke(tool_call["args"], config)
    
        result.append({"role": "tool", 
                       "content": observation, 
                       "name": tool_call["name"], 
                       "tool_call_id": tool_call["id"]})
        
        
        # 각 도구별로 실행 결과를 변수에 저장 (for loop 종료 후 라우팅 결정에 사용)
        
        # 질문 도구가 호출된 경우 즉시 사용자에게 질문 반환
        if tool_call["name"] == "Question":
            question_obj = cast(Question, observation)   
            result.append({"role": "assistant", "content": question_obj.question})    # AI가 사용자에게 질문하기 떄문에 assistant 역할로 추가
            
            return Command(goto=END, update={"messages": result})
        
        elif tool_call["name"] == "FinishReport":
            result.append({"role": "user", "content": "Report is finish"})            # 시스템이 LLM에게 지시/안내하는 메세지이므로 user 역할로 추가
            return Command(goto=END, update={"messages": result})

        elif tool_call["name"] == "Sections":
            sections_list = cast(Sections, observation).sections
        
        # 제목에 H1 Heading을 붙임 
        elif tool_call["name"] == "Introduction":
            observation = cast(Introduction, observation)
            if not observation.content.startswith("# "):
                intro_content = f"# {observation.name} \n\n {observation.content}"
            else:
                intro_content = observation.content
        
        # 결론은 H2 Heading 붙임
        elif tool_call["name"] == "Conclusion":
            observation = cast(Conclusion, observation)
            if not observation.content.startswith("## "):
                conclusion_content = f"## {observation.name}\n\n{observation.content}"
            else:
                conclusion_content = observation.content        

        elif tool_call["name"] in search_tool_names and configurable.include_source_str:
            source_str += cast(str, observation)            
    
    # 모든 도구 호출 후 다음 Step 결정 
    if sections_list:
        # 각 섹션을 Research Agent가 작성해야 하므로 research_team으로 라우팅
        send_list = []
        for s in sections_list:
            send_list.append(Send("research_team", {'section': s}))
        return Command(goto=send_list, update = {"messages": result})
    
    # 서론이 작성되었고 결론이 아직 작성되지 않은 경우 결론 작성 지시
    elif intro_content:
        result.append({"role": "user", "content": "Introduction written. Now write a conclusion section."})  # 시스템이 LLM에게 지시/안내하는 메세지이므로 user 역할로 추가
        
        # 상태 업데이트(최초에 final_report에는 빈 문자열이 들어있음)
        state_update = {
            "final_report": intro_content,
            "messages": result
        }

    elif conclusion_content:
        # 최종 보고서 조립, 모든 섹션을 가져와서 올바른 순서로 결합 (서론 + 본문 섹션 + 결론)
        intro = state.get("final_report", "")
        
        completed = []
        for s in state["completed_sections"]:
            completed.append(s.content)
            
        body_sections = "\n\n".join(completed)
        
        complete_report = f"{intro}\n\n{body_sections}\n\n{conclusion_content}"
        
        result.append({"role": "user", "content": "Report is now complete with introduction, body sections, and conclusion."})

        # 최종 보고서 상태 업데이트 (final_report 갱신(기존엔 intro만 들어있음))
        state_update = {
            "final_report": complete_report,
            "messages": result,
        }    

    else:
        state_update = {"messages": result}

    # 검색 결과 문자열 포함 (평가용)
    if configurable.include_source_str and source_str:
        state_update["source_str"] = source_str

    return Command(goto="supervisor", update=state_update)

async def supervisor_continue(state: ReportState) -> str:
    """
    Supervisor 실행 후 도구 호출을 했는지 여부에 따라 다음 노드를 결정하는 라우팅 함수.

    LLM이 tool_calls를 생성했으면 supervisor_tools로 이동하여 도구를 실행.
    tool_calls가 없으면 그래프를 종료 (END).
    """
    
    messages = state["messages"]
    
    # 마지막 메세지에 도구 호출이 없으면 종료
    if not messages[-1].tool_calls:
        return END
    
    return "supervisor_tools"


async def research_agent(state: SectionState, config: RunnableConfig):
    """
    Research Agent LLM이 섹션 연구를 위한 다음 행동을 결정

    - Supervisor로부터 할당받은 특정 섹션을 연구하고 작성하는 연구원 역할
    - LLM이 도구를 선택하여 다음 행동 결정 (검색, 섹션 작성, 연구 완료)
    - ReAct 패턴의 Reason 단계 (추론 및 도구 선택)

    Args:
        state: SectionState
            - section: supervisor가 할당한 섹션 이름 (예: "AI Safety Overview")
            - messages: 이전 대화 히스토리 (ToolMessage 포함)
        config: RunnableConfig (모델 설정 및 API 키)

    Returns:
        {"messages": [AIMessage(tool_calls=[...])]}
            - LLM이 결정한 도구 호출 정보
            - research_agent_should_continue로 전달되어 다음 노드 결정
    """
    
    # 설정 가져오기
    configurable = Configuration.from_runnable_config(config)
    research_model = configurable.researcher_model
    
    research_llm = init_chat_model(research_model)
    
    # tools list 
    research_tool_list = await get_research_tools(config)
    
    # research LLM 프롬프트 
    system_prompt = RESEARCH_INSTRUCTIONS.format(
        section_description=state['section'],
        number_of_queries = configurable.number_of_queries,
        today=get_today()
    )


    llm_with_tools = research_llm.bind_tools(
        research_tool_list,
        parallel_tool_calls=False,
        tool_choice="any",  # 최소 하나의 도구 실행 
        )
    
    # 첫 실행시 초기 user 메시지 생성
    messages = state.get("messages", [])
    if not messages:
        messages = [{"role": "user", "content": f"Please research and write the section: {state['section']}"}]
    
    
    # LLM이 어떤 도구를 사용할지 결정
    response = await llm_with_tools.ainvoke([{"role": "system", "content": system_prompt}] + messages)

        
    return {"messages": [response]}    




state = {
    # "messages": [{"role": "user", "content": "딥러닝의 역사에 대해서 조사"}],
    # "messages": ["딥러닝의 역사에 대해서 조사"],
    "messages": [HumanMessage(content="딥러닝의 역사에 대해서 조사")],
    "sections": [],  # 섹션 목록
    "completed_sections": [],  # 완성된 섹션들
    "final_report": "",  # 최종 보고서
    "source_str": ""  # 검색 소스
}

config = {"configurable": {
                           "search_api": "tavily",
                           }}



asyncio.run(supervisor(state, config))





# # Research agent workflow
# research_builder = StateGraph(SectionState, output=SectionOutputState, config_schema=Configuration)
# research_builder.add_node("research_agent", research_agent)

# # Edge
# research_builder.add_edge(START, "research_agent")

# # Supervisor workflow
# supervisor_builder = StateGraph(ReportState, input=MessagesState, output=ReportStateOutput, config_schema=Configuration)
# supervisor_builder.add_node("supervisor", supervisor)
# supervisor_builder.add_node("supervisor_tools", supervisor_tools)

# # Edge
# supervisor_builder.add_edge(START, "supervisor")
# supervisor_builder.add_conditional_edges("supervisor", supervisor_continue, ["supervisor_tools", END])  # supervisor의 결과가 tools이 없이 supervisor_tools로 가면 에러가 발생하기 때문에 conditional_edge 사용하여 supervisor_continue를 통해 END로 보냄 
# supervisor_builder.add_node("research_team", research_builder.compile())
# supervisor_builder.add_edge("research_team", "supervisor")


# app = supervisor_builder.compile()




# import requests 

# response = requests.get(
#   "https://api.tavily.com/usage",
#   headers={"Authorization": "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"}
# )

# response.json()