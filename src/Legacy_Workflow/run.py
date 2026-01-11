# from rich import print
# from rich.panel import Panel
# from rich.console import Console, Group
# from rich.spinner import Spinner
# from rich.live import Live
# from rich.text import Text
# from rich import print
# import time

# from src.Legacy_Workflow.state import (
#     SearchQuery
    
# )

# import time 

# console = Console()


# console.print(Panel(f"[bold blue]주제: {'인공지능에대해서 심층리서치'}[/bold blue]", title="📋 보고서 계획 생성"))

# console = Console()
# log = Text()

# spinner = Spinner("dots", text="[bold green]사용자가 원하는 토픽에 관해 조사하기 위해 검색 쿼리를 생성 중 입니다.")

# with Live(Group(spinner, log), console=console, refresh_per_second=10):
    
#     time.sleep(2)
    
#     queries = ["AI 역사", "AI 응용분야", "AI 미래 전망"]
#     for q in queries:
#         log.append(f"\n- {q}")
#         time.sleep(0.7)

# spinner = Spinner("dots", text="[bold green]생성된 검색 쿼리로 보고서 계획을 위한 웹 검색을 수행 중입니다...")

# with Live(Group(spinner, log), console=console, refresh_per_second=10):
    
#     time.sleep(2)
    


from rich.console import Console, Group
from rich.spinner import Spinner
from rich.live import Live
from rich.text import Text
import time

console = Console()

header = Text()
body = Text()

spinner = Spinner("dots", text="검색 쿼리 생성 중...")

def set_stage(text):
    global header
    header = Text(text)

def clear_body():
    global body
    body = Text()

with Live(Group(spinner, header, body), console=console, refresh_per_second=12) as live:

    def refresh():
        live.update(Group(spinner, header, body))

    # 1단계
    set_stage("[bold green]검색 쿼리 생성 중...[/bold green]")
    clear_body()
    refresh()

    queries = ["AI 역사", "AI 응용분야", "AI 미래 전망"]
    for q in queries:
        body.append(f"\n- {q}")
        refresh()
        time.sleep(0.7)

    # 요약
    clear_body()
    body.append("[green]✔ 검색 쿼리 3개 생성 완료[/green]")
    refresh()

    # 2단계
    spinner.text = "웹 검색 중..."
    set_stage("[bold cyan]웹 검색 중...[/bold cyan]")
    refresh()
    time.sleep(2)

    # 3단계
    spinner.text = "분석 중..."
    set_stage("[bold magenta]분석 중...[/bold magenta]")
    refresh()
    time.sleep(2)



# from rich.console import Console, Group
# from rich.spinner import Spinner
# from rich.live import Live
# from rich.text import Text
# from rich import print
# import time

# from langchain.chat_models import init_chat_model

# llm = init_chat_model(model="openai:gpt-5-mini")

# console = Console()
# log = Text()
# spinner = Spinner("dots", text="검색 쿼리 생성 중...")

# with Live(Group(spinner, log), console=console, refresh_per_second=20):

#     for chunk in llm.stream(
#         "AI에 대해 검색할 쿼리 3개만 리스트로 만들어줘."
#     ):
#         if chunk.content:
#             log.append(chunk.content)   # ← 일부러 토큰 단위로 append