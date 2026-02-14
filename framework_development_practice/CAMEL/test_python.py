import os

from camel.models import ModelFactory
from camel.societies import RolePlaying
from camel.types import ModelPlatformType
from camel.utils import print_text_animated
from colorama import Fore



model=ModelFactory.create(
    model_platform=ModelPlatformType.DEEPSEEK,
    model_type="deepseek-chat",
    url=os.getenv("LLM_BASE_URL"),
    api_key=os.getenv("LLM_API_KEY"),
)
task="""
创建一本关于python基础的书
要求：
内容科学严谨，语义通俗易懂
80-100字
"""

print(Fore.YELLOW+f"任务{task}")

role_play_session=RolePlaying(
    model=model,
    assistant_role_name="多年经验的python程序员",
    user_role_name="作家",
    task_prompt=task,
    with_task_specify=False,
)

print(Fore.CYAN+f"{role_play_session.task_prompt}")


chat_turn_limit,n=30,0
init_chat = role_play_session.init_chat()
while n<chat_turn_limit:
    n+=1
    assistant_response,user_response = role_play_session.step(init_chat)

    if assistant_response.msg is  None or user_response.msg is  None:
        break
    print_text_animated(Fore.BLUE+f"作家\n\n{user_response.msg.content}\n")
    print_text_animated(Fore.RED+f"python\n\n{assistant_response.msg.content}\n")

    if "<CAMEL_TASK_DONE>" in user_response.msg.content or "<CAMEL_TASK_DONE>" in user_response.msg.content:
        print(Fore.MAGENTA+"制作完成")
        break
    init_chat=assistant_response.msg
print(Fore.BLUE+f"有{n}轮对话")

