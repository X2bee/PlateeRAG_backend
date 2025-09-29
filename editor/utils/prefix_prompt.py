from datetime import datetime
import pytz

default_prefix_prompt_content = """질문에 대해서 한국어로 답변하세요.\n"""

def get_current_time_kr():
    seoul_tz = pytz.timezone('Asia/Seoul')
    current_time = datetime.now(seoul_tz)
    return current_time.strftime("현재 시간은 %Y년 %m월 %d일 %H시 %M분 입니다.")

def get_prefix_prompt():
    return f"{get_current_time_kr()}\n{default_prefix_prompt_content}"

# 하위 호환성을 위한 변수 (deprecated)
prefix_prompt = get_prefix_prompt()
