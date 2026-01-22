from typing import Annotated, TypedDict
from langchain.messages import HumanMessage, AIMessage, ToolMessage
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from rich import print

# Tool 정의
@tool
def get_weather(city: str) -> str:
    """도시 이름을 받아서 현재 날씨를 반환"""
    return f"{city}의 현재 날씨는 맑고 25도입니다."


# State 정의
class AgentState(TypedDict):
    messages: Annotated[list, add_messages]


# LLM 설정
tools = [get_weather]

reasoning = {
    "effort": "low",  # 'low', 'medium', or 'high'
    "summary": "auto",  # 'detailed', 'auto', or None
}

llm = init_chat_model(model="gpt-4o", temperature=0, reasoning=reasoning)
llm_with_tools = llm.bind_tools(tools)

response = llm.invoke('앵무새의 깃털 색깔은 왜 다양할까?')
print(response)

    


for chunk in llm.stream("앵무새의 깃털 색깔은 왜 다양할까?"):
    reasoning_steps = [r for r in chunk.content_blocks if r["type"] == "reasoning"]
    
    print(reasoning_steps)
    
    break
        
    # reasoning_steps = [r for r in chunk.content_blocks if r["type"] == "reasoning"]
    # print(reasoning_steps if reasoning_steps else chunk.text, end='', flush=True)
    
    
    
    
    
# Agent 노드
def agent(state: AgentState):
    messages = state["messages"]
    # response = llm_with_tools.invoke(messages)
    system_prompt = """당신은 유용한 비서입니다. 사용자의 질문에 답변하고 필요시 도구를 사용하세요.
    
                        도구를 선택하기 전에 사용자에게 어떻게 결과를 제공할건지에 대해 간략히 계획을 세우고 그 계획과 
                        
                        선택한 도구에 대한 설명을 사용자에게 제공하세요"""
    
    response = llm_with_tools.invoke(([{"role": "system", "content": system_prompt}]) + messages)
    
    return {"messages": [response]}


def tool_node(state: AgentState):
    
    # 각 tool 실행
    tool_messages = []
    for tool_call in state["messages"][-1].tool_calls:
        # tool 찾기
        tool = None
        for t in tools:
            if t.name == tool_call["name"]:
                tool = t
                break
            
        # tool 실행
        result = tool.invoke(tool_call["args"])
        
        # ToolMessage 생성
        tool_messages.append(
            ToolMessage(
                content=result,
                tool_call_id=tool_call["id"]
            )
        )
    
    return {"messages": tool_messages}


# Tool 실행 여부 결정
def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    
    # Tool call이 있으면 tools 노드로, 없으면 종료
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return END


# Graph 생성
workflow = StateGraph(AgentState)

# 노드 추가
workflow.add_node("agent", agent)
# workflow.add_node("tools", ToolNode(tools))
workflow.add_node("tools", tool_node)

# Edge 추가
workflow.add_edge(START, "agent")
# workflow.add_conditional_edges("agent", should_continue, ["tools", END])
# workflow.add_edge("tools", "agent")

# 컴파일
app = workflow.compile()


response = app.invoke({"messages": "서울의 날씨 알려줘"})

print(response)
