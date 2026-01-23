from datetime import datetime

def get_today() -> str:
    """현재 날짜를 "2025-10-10" 형식의 문자열로 반환하는 함수."""
    return datetime.now().strftime("%Y-%m-%d")
