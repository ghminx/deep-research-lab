from datetime import datetime
from langchain.tools import tool
from langchain_core.messages import (
    MessageLikeRepresentation,
    filter_messages
)

def get_today() -> str:
    """현재 날짜를 "2025-10-10" 형식의 문자열로 반환하는 함수."""
    return datetime.now().strftime("%Y-%m-%d")

@tool(description="Strategic reflection tool for research planning")
def think_tool(reflection: str) -> str:
    """Tool for strategic reflection on research progress and decision-making.

    Use this tool after each search to analyze results and plan next steps systematically.
    This creates a deliberate pause in the research workflow for quality decision-making.

    When to use:
    - After receiving search results: What key information did I find?
    - Before deciding next steps: Do I have enough to answer comprehensively?
    - When assessing research gaps: What specific information am I still missing?
    - Before concluding research: Can I provide a complete answer now?

    Reflection should address:
    1. Analysis of current findings - What concrete information have I gathered?
    2. Gap assessment - What crucial information is still missing?
    3. Quality evaluation - Do I have sufficient evidence/examples for a good answer?
    4. Strategic decision - Should I continue searching or provide my answer?

    Args:
        reflection: Your detailed reflection on research progress, findings, gaps, and next steps

    Returns:
        Confirmation that reflection was recorded for decision-making
    """
    return f"Reflection recorded: {reflection}"



def get_notes_from_tool_calls(messages: list[MessageLikeRepresentation]):
    """도구 호출 메세지에서 노트 추출"""
    return [tool_msg.content for tool_msg in filter_messages(messages, include_types="tool")]