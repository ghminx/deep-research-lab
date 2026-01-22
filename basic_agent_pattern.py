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
llm = init_chat_model(model="gpt-5-mini", temperature=0)
llm_with_tools = llm.bind_tools(tools)


# Agent 노드
def agent(state: AgentState):
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
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
workflow.add_conditional_edges("agent", should_continue, ["tools", END])
workflow.add_edge("tools", "agent")

# 컴파일
app = workflow.compile()


# 실행 예시
if __name__ == "__main__":
    # 스트리밍으로 실행
    for step in app.stream(
        {"messages": [HumanMessage(content="서울의 날씨를 알려줘")]},
        stream_mode="updates"
    ):
        print(step)
