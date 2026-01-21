from dataclasses import dataclass

from langchain.chat_models import init_chat_model
from langgraph.graph import StateGraph, START, MessagesState
from langchain.messages import SystemMessage, HumanMessage

class MyState(MessagesState):
    pass

model = init_chat_model(model="gpt-4o-mini")

def call_model(state: MyState):
    # state["messages"]에 담긴 사용자 메시지와 시스템 메시지를 합쳐서 전달
    system_msg = SystemMessage(content="You are a assistant AI for human")
    model_response = model.invoke([system_msg] + state["messages"])
    
    # 반드시 딕셔너리 형태로 반환하여 상태를 업데이트해야 함
    return {"messages": [model_response]}

graph = (
    StateGraph(MyState)
    .add_node("call_model", call_model) # 노드 이름 명시
    .add_edge(START, "call_model")
    .compile()
)

# 입력 데이터는 {"messages": [HumanMessage(...)]} 형식이어야 함
for message_chunk, metadata in graph.stream(
    {"messages": [HumanMessage(content="gpt에 대해서 알려줘")]},
    stream_mode="updates",  
):
    if message_chunk.content:
        print(message_chunk.content, end="", flush=True)