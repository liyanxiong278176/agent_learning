import asyncio
import os
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv

load_dotenv()

model_client = OpenAIChatCompletionClient(
    model="deepseek-chat",
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com/v1",
    model_info={
        "function_calling": True,
        "max_tokens": 4096,
        "context_length": 32768,
        "vision": False,
        "json_output": True,
        "family": "deepseek",
        "structured_output": True,
    }
)

primary_agent = AssistantAgent(
    "primary",
    model_client=model_client,
    system_message="你是一个文学家",
)



user_proxy = UserProxyAgent("user_proxy", input_func=input)

text_termination = TextMentionTermination("APPROVE")

team = RoundRobinGroupChat([primary_agent, user_proxy], termination_condition=text_termination)

async def main():
    await Console(team.run_stream(task="编写一首唐诗"))

if __name__ == "__main__":
    asyncio.run(main())