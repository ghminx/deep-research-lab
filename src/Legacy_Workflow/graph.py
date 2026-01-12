from rich import print
from rich.console import Console
from rich.panel import Panel




import asyncio

from typing import Literal


from langchain.chat_models import init_chat_model
from langchain_core.runnables import RunnableConfig
from langchain.messages import SystemMessage, HumanMessage

from langgraph.constants import Send
from langgraph.graph import START, END, StateGraph
from langgraph.types import interrupt, Command

from src.Legacy_Workflow.state import (
    ReportState,
    ReportStateInput,
    ReportStateOutput,
    SectionState,
    Section,
    SectionOutputState,
    
    Queries,
    Sections,
    Feedback,
    
    
)

from src.Legacy_Workflow.prompts import (
    report_planner_query_writer_instructions,
    report_planner_instructions,
    query_writer_instructions,
    
    section_writer_inputs,
    section_writer_instructions,
    
    
    section_grader_instructions,
    
    
    final_section_writer_instructions
)


from src.Legacy_Workflow.utils import (
    get_search_params,
    get_today,
    run_search,
    format_sections,
)

from src.Legacy_Workflow.config import Configuration


console = Console()

async def generate_report_plan(state: ReportState, config: RunnableConfig):
    """섹션들로 구성된 보고서 계획을 생성하는 노드 
    
    1. 보고서 구조와 파라미터 설정을 가져옴 
    2. 계획을 수립하기 위해 검색 쿼리를 생성 
    3. 생성한 검색쿼리들로 웹 검색 수행 
    4. LLM을 통해 섹션(서론, 본론, 결론)들로 구성된 구조화된 계획을 생성 

    Args:
        state (ReportState): 그래프 상태(Topic) 
        config (RunnableConfig): 모델, 검색 API등의 설정 정보 
        
    Returns:
        dict: 섹션들로 구성된 보고서 계획
    """
    
    # 입력값 - 주제 
    topic = state["topic"]
    
    console.print(Panel(f"[bold blue]주제: {topic}[/bold blue]", title="📋 보고서 계획 생성"))
    
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
    

    # 검색 쿼리 생성용 LLM 정의
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
    search_message = "Generate search queries that will help with planning the sections of the report."
    
    results = await structured_llm.ainvoke([SystemMessage(content=search_query_prompt),
                                    HumanMessage(content=search_message)])
    query_list = [query.search_query for query in results.queries]
    
    # 보고서 계획을 위한 웹 검색 수행
    web_source = await run_search(search_api, query_list, web_param_filter)


    # 섹션 계획 생성용 LLM 정의
    planner_provider = configurable.planner_provider
    planner_model_name= configurable.planner_model
    planner_model_kwargs = configurable.planner_model_kwargs or {}
    planner_llm = init_chat_model(model=planner_model_name, model_provider=planner_provider, max_tokens = 10000, model_kwargs=planner_model_kwargs)
    structured_llm = planner_llm.with_structured_output(Sections)
    
    # 보고서 섹션 계획 생성 프롬프트 포맷팅
    system_instructions_sections = report_planner_instructions.format(
        topic=topic, 
        report_organization=report_structure, 
        context=web_source, 
        feedback=feedback
    )
    
    # 보고서 섹션 계획 생성 
    planner_message = """Generate the sections of the report. Your response must include a 'sections' field containing a list of sections. 
                        Each section must have: name, description, research, and content fields."""
    report_sections = await structured_llm.ainvoke([SystemMessage(content=system_instructions_sections),
                                             HumanMessage(content=planner_message)])

    sections = report_sections.sections
    
    # 노드의 출력값은 딕셔너리 형태여야 하고 state에 업데이트 됨 
    return {"sections": sections}


# 휴먼 피드백 
# Command:  LangGraph에서 노드의 다음 행동을 지시하는 객체
def human_feedback(state: ReportState, config: RunnableConfig) -> Command[Literal["generate_report_plan","build_section_with_web_research"]]:
    """보고서 계획에 대한 사용자 피드백을 받고 다음 단계로 라우팅

    이 노드의 역할:
    1. 현재 보고서 계획을 사용자가 검토할 수 있게 포맷팅
    2. interrupt를 통해 피드백 받음
    3. 다음 중 하나로 라우팅:
    - 승인 → 섹션 작성으로 이동
    - 피드백 제공 → 계획 재생성으로 이동

    Args:
        state: 검토할 섹션들이 담긴 그래프 상태
        config: 워크플로우 설정
        
    Returns:
        계획 재생성 또는 섹션 작성을 시작하는 Command
    """

    # sections 가져오기 
    topic = state["topic"]
    sections = state["sections"]
    

    section_list = []

    for section in sections:
        text = f"Section: {section.name}\n"
        text += f"Description: {section.description}\n"
        text += f"Research needed: {'Yes' if section.research else 'No'}"
        
        section_list.append(text)
        
    sections_str = '\n\n'.join(section_list)

    # 인터럽트에서 보고서 계획에 대한 피드백 받기
    interrupt_message = f"""
    출력된 보고서 작성 계획을 검토해주세요. 
                        
    \n\n{sections_str}\n\n


    출력된 보고서 작성 계획을 승인하려면 'true'를 입력하세요.
    수정이 필요하면 피드백을 입력하세요:"
    """
    
    feedback = interrupt(interrupt_message)
    
    # 사용자가 보고서 계획 승인 여부 확인
    if isinstance(feedback, bool) and feedback is True:
        
        send_list = []
        for s in sections:
            if s.research:    
                send_list.append(
                    Send("build_section_with_web_research",      # Send("node_name", {"key": "value"})
                            {
                            "topic": topic,
                            "section": s,
                            "search_iterations": 0  # 검색 반복 횟수 초기값
                            }
                        ))
                
        return Command(goto=send_list)
    
    # 사용자가 피드백을 입력한 경우 generate_report_plan 노드로 이동
    elif isinstance(feedback, str):
        return Command(goto="generate_report_plan", 
                       update={"feedback_on_report_plan": [feedback]})  # feedback_on_report_plan 상태를 업데이트
    else:
        raise TypeError(f"{type(feedback)}의 형식은 지원하지 않습니다. 문자열로 된 피드백이나 'true'를 입력해주세요.")


async def generate_queries(state: SectionState, config: RunnableConfig):
    """
    특정 섹션을 리서치하기 위한 검색 쿼리를 생성.

    이 노드는 LLM을 사용하여  
    섹션의 주제와 설명을 기반으로 목표 지향적인(정밀한) 검색 쿼리를 생성한다.

    Args:
        state: 섹션에 대한 정보가 들어 있는 현재 상태
        config: 생성할 검색 쿼리 개수를 포함한 설정 정보

    Returns:
        생성된 검색 쿼리를 담은 Dict(딕셔너리)
    """    
    
    # state 값들 가져오기
    topic = state["topic"]
    section = state["section"]
    
    # 설정값 가져오기
    configuration = Configuration.from_runnable_config(config)
    number_of_queries = configuration.number_of_queries

    # 검색 쿼리 생성용 LLM 정의
    writer_provider = configuration.writer_provider
    writer_model_name = configuration.writer_model
    writer_model_kwargs = configuration.writer_model_kwargs or {}
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs)
    structured_llm = writer_model.with_structured_output(Queries)  # 구조화된 출력 형태 지정    
    
    # 검색 쿼리 생성 프롬프트
    system_instructions = query_writer_instructions.format(
        topic=topic,
        section_topic = section.description,
        number_of_queries=number_of_queries,
        today = get_today()
    )
    
    
    # 검색 쿼리 생성
    search_message = "Generate search queries on the provided topic."
    queries = await structured_llm.ainvoke([SystemMessage(content=system_instructions),
                                    HumanMessage(content=search_message)])

    return {'search_queries': queries.queries}

async def search_web(state: SectionState, config: RunnableConfig):
    """섹션별(본문1, 본문2 등) 웹 검색을 실행.
    
    이 노드는:
    1. 생성된 쿼리들을 가져온다
    2. 설정된 검색 API를 사용하여 검색을 실행한다
    3. 결과를 사용 가능한 컨텍스트로 포맷팅한다
    
    Args:
        state: 검색 쿼리가 포함된 현재 상태
        config: 검색 API 설정
        
    Returns:
        검색 결과와 업데이트된 반복 횟수를 담은 Dict
    """
    
    # state 가져오기 
    search_queries = state['search_queries']
    
    # 설정값 가져오기
    configuration = Configuration.from_runnable_config(config)
    search_api = configuration.search_api
    search_api_config = configuration.search_api_config or {}
    web_param_filter = get_search_params(search_api, search_api_config)

    # Pydantic 모델에서 쿼리 문자열 리스트로 변환
    query_list = [query.search_query for query in search_queries]
    
    # 웹 검색 실행 
    web_source = await run_search(search_api, query_list, web_param_filter)
    
    return {"source_str": web_source, "search_iterations": state["search_iterations"] + 1}


async def write_section(state: SectionState, config: RunnableConfig) -> Command[Literal[END, "search_web"]]:
    """보고서의 섹션을 작성하고 추가 리서치가 필요한지 평가.
    
    이 노드는:
    1. 검색 결과를 사용하여 섹션 내용을 작성한다
    2. 섹션의 품질을 평가한다
    3. 다음 중 하나를 수행한다:
       - 품질이 통과하면 섹션을 완료한다
       - 품질이 실패하면 추가 리서치를 트리거한다
    
    Args:
        state: 검색 결과와 섹션 정보가 포함된 현재 상태
        config: 작성 및 평가를 위한 설정
        
    Returns:
        섹션 완료 또는 추가 리서치를 위한 Command
    """
    
    # state 가져오기
    topic = state["topic"]
    section = state["section"]
    source_str = state["source_str"]
    
    # 설정값 가져오기
    configuration = Configuration.from_runnable_config(config)

    # Section 작성 LLM 정의 
    writer_model_name = configuration.writer_model
    writer_provider = configuration.writer_provider
    writer_model_kwargs = configuration.writer_model_kwargs or {}
    writer_llm = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs)
    
    # Section 작성 프롬프트
    section_writer_prompt = section_writer_inputs.format(
        topic=topic,
        section_name = section.name,
        section_topic = section.description,
        context=source_str,
        section_content=section.content
    )   
    
    # Section 작성
    section_content = await writer_llm.ainvoke([SystemMessage(content=section_writer_instructions),
                                                HumanMessage(content=section_writer_prompt)])
    
    # 생성된 섹션 콘텐츠를 섹션에 필드에 업데이트 
    section.content = section_content.content

    # Section 평가 LLM 정의
    grade_model_name = configuration.planner_model
    grade_model_provider = configuration.planner_provider
    grade_model_kwargs = configuration.planner_model_kwargs or {}
    grade_model = init_chat_model(model=grade_model_name, model_provider=grade_model_provider, model_kwargs=grade_model_kwargs)
    structured_llm = grade_model.with_structured_output(Feedback)

    # Section 평가 프롬프트
    section_grader_message = ("Grade the report and consider follow-up questions for missing information. "
                              "If the grade is 'pass', return empty strings for all follow-up queries. "
                              "If the grade is 'fail', provide specific search queries to gather missing information.")
    
    section_grader_prompt = section_grader_instructions.format(topic=topic, 
                                                                               section_topic=section.description,
                                                                               section=section.content, 
                                                                               number_of_follow_up_queries=configuration.number_of_queries)

    # Section 평가 
    feedback = await structured_llm.ainvoke([SystemMessage(content=section_grader_prompt),
                                             HumanMessage(content=section_grader_message)])
    
    # 평가 결과에 따라 다음 단계 결정
    if feedback.grade == "pass" or state['search_iterations'] >= configuration.max_search_depth:
        update = {"completed_sections": [section]}
        
        console.print(Panel(f"[green bold]평가 결과: pass[/green bold]"))
        
        if configuration.include_source_str:
            update["source_str"] = state["source_str"]
            
        return Command(goto=END, update=update)
    
    else:
        console.print(Panel(f"[green bold]재검색 필요: {feedback.grade}[/green bold]"))
        
        return Command(goto="search_web",
                       update={"search_queries": feedback.follow_up_queries, "section": section})
    
def gather_completed_sections(state: ReportState):
    """완료된 섹션들을 최종 섹션 작성을 위한 컨텍스트로 포맷팅

    완료된 모든 리서치 섹션들을 가져와서 문자열로 포맷팅

    Args:
        state: 완료된 섹션들이 담긴 현재 상태
        
    Returns:
        포맷팅된 섹션들을 컨텍스트로 담은 딕셔너리
    """
    
    completed_sections = state["completed_sections"]
    
    completed_report_sections = format_sections(completed_sections)
    
    return {"report_sections_from_research": completed_report_sections}
    
def initiate_final_section_writing(state: ReportState):
    """
    리서치가 필요 없는 섹션(서론, 결론)을 write_final_sections로 병렬 전송합니다.
    
    이 함수는 add_conditional_edges의 라우팅 함수로 사용됩니다.
    - 대안: add_node + Command(goto=[Send(...)]) 방식도 가능 (동작 동일, 노드로 보임)
    
    Args:
        state: 모든 섹션과 리서치 컨텍스트가 포함된 현재 상태

    Returns:
        병렬 섹션 작성을 위한 Send 명령 리스트
    """

    send_list = []
    for s in state['sections']:
        if not s.research:
            send_list.append(
                Send("write_final_sections", 
                    {
                        "topic": state["topic"],
                        "section": s,
                        "report_sections_from_research": state["report_sections_from_research"]
                    }
                )
            )
            
    return send_list

async def write_final_sections(state: SectionState, config: RunnableConfig):
    """리서치가 필요 없는 섹션을 완료된 섹션들을 컨텍스트로 사용하여 작성
    
    이 노드는 직접적인 리서치 대신 리서치된 섹션들을 기반으로 
    서론, 결론이나 요약 같은 섹션을 처리

    Args:
        state: 완료된 섹션들이 컨텍스트로 포함된 현재 상태
        config: 작성 모델을 위한 설정

    Returns:
        새로 작성된 섹션이 포함된 딕셔너리

    """

    # state 값들 가져오기
    topic = state["topic"]
    section = state["section"]
    completed_report_sections = state["report_sections_from_research"]
    
    # 설정값 가져오기
    configurable = Configuration.from_runnable_config(config)
    
    # 최종 섹션 작성용 LLM 정의
    writer_provider = configurable.writer_provider
    writer_model_name = configurable.writer_model
    writer_model_kwargs = configurable.writer_model_kwargs or {}
    writer_model = init_chat_model(model=writer_model_name, model_provider=writer_provider, model_kwargs=writer_model_kwargs) 
    
    # 최종 섹션 작성 프롬프트
    system_instructions = final_section_writer_instructions.format(topic=topic, 
                                                                   section_name=section.name, 
                                                                   section_topic=section.description, 
                                                                   context=completed_report_sections)

    section_content = await writer_model.ainvoke([SystemMessage(content=system_instructions),
                                           HumanMessage(content="Generate a report section based on the provided sources.")])
    
    # 생성된 섹션 콘텐츠를 섹션에 필드에 업데이트 
    section.content = section_content.content

    return {"completed_sections": [section]}

# 서브 그래프 정의     
section_bulder = StateGraph(SectionState, output=SectionOutputState)
section_bulder.add_node("generate_queries", generate_queries)
section_bulder.add_node("search_web", search_web)
section_bulder.add_node("write_section", write_section)

# 서브 그래프 엣지 추가
section_bulder.add_edge(START, "generate_queries")
section_bulder.add_edge("generate_queries", "search_web")
section_bulder.add_edge("search_web", "write_section")

# 메인 그래프 정의
# builder의 각 노드의 모든 출력 결과는 ReportState에 병합됨
builder = StateGraph(ReportState, input=ReportStateInput, output=ReportStateOutput, config_schema=Configuration)
builder.add_node("generate_report_plan", generate_report_plan)
builder.add_node("human_feedback", human_feedback)
builder.add_node("build_section_with_web_research", section_bulder.compile())
builder.add_node("gather_completed_sections", gather_completed_sections)
builder.add_node("write_final_sections", write_final_sections)



# 메인 그래프 엣지 추가
builder.add_edge(START, "generate_report_plan")
builder.add_edge("generate_report_plan", "human_feedback")
builder.add_edge("build_section_with_web_research", "gather_completed_sections")
builder.add_conditional_edges("gather_completed_sections", initiate_final_section_writing, ["write_final_sections"])





# graph = builder.compile()


