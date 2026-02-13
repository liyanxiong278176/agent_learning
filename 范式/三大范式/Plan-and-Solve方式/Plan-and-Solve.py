import ast

from 范式.执行器.Plan_and_Solve_executor import Executor
from 范式.llm_client import HelloAgentsLLM

PLANNER_PROMPT_TEMPLATE = """
你是一个顶级的AI规划专家。你的任务是将用户提出的复杂问题分解成一个由多个简单步骤组成的行动计划。
请确保计划中的每个步骤都是一个独立的、可执行的子任务，并且严格按照逻辑顺序排列。
你的输出必须是一个Python列表，其中每个元素都是一个描述子任务的字符串。

问题: {question}

请严格按照以下格式输出你的计划,```python与```作为前后缀是必要的:
```python
["步骤1", "步骤2", "步骤3", ...]
```
"""


# 假定 llm_client.py 中的 HelloAgentsLLM 类已经定义好
# from llm_client import HelloAgentsLLM

class Planner:
    def __init__(self, llm_client):
        self.llm_client = llm_client

    def plan(self, question: str) -> list[str]:
        """
        根据用户问题生成一个行动计划。
        """
        prompt = PLANNER_PROMPT_TEMPLATE.format(question=question)

        # 为了生成计划，我们构建一个简单的消息列表
        messages = [{"role": "user", "content": prompt}]

        print("--- 正在生成计划 ---")
        # 使用流式输出来获取完整的计划
        response_text = self.llm_client.think(messages=messages) or ""

        print(f"✅ 计划已生成:\n{response_text}")

        # 解析LLM输出的列表字符串
        try:
            # 找到```python和```之间的内容
            plan_str = response_text.split("```python")[1].split("```")[0].strip()
            # 使用ast.literal_eval来安全地执行字符串，将其转换为Python列表
            plan = ast.literal_eval(plan_str)
            return plan if isinstance(plan, list) else []
        except (ValueError, SyntaxError, IndexError) as e:
            print(f"❌ 解析计划时出错: {e}")
            print(f"原始响应: {response_text}")
            return []
        except Exception as e:
            print(f"❌ 解析计划时发生未知错误: {e}")
            return []


class PlanAndSolveAgent:
    def __init__(self, llm_client):
        """
        初始化智能体，同时创建规划器和执行器实例。
        """
        self.llm_client = llm_client
        self.planner = Planner(self.llm_client)
        self.executor = Executor(self.llm_client)

    def run(self, question: str):
        """
        运行智能体的完整流程:先规划，后执行。
        """
        print(f"\n--- 开始处理问题 ---\n问题: {question}")

        # 1. 调用规划器生成计划
        plan = self.planner.plan(question)

        # 检查计划是否成功生成
        if not plan:
            print("\n--- 任务终止 --- \n无法生成有效的行动计划。")
            return

        # 2. 调用执行器执行计划
        final_answer = self.executor.execute(question, plan)

        print(f"\n--- 任务完成 ---\n最终答案: {final_answer}")
        return final_answer


# 在文件末尾添加以下 main 函数

def main():
    """
    Plan-and-Solve 智能体主函数 - 苹果计算问题实例
    """
    print("=" * 70)
    print("🍎 Plan-and-Solve 智能体 - 苹果计算问题实例")
    print("=" * 70)

    # 测试问题 - 你的苹果计算问题
    test_question = "一个水果店周一卖出了15个苹果。周二卖出的苹果数量是周一的两倍。周三卖出的数量比周二少了5个。请问这三天总共卖出了多少个苹果？"
    # 首先，我们需要导入 LLM 客户端

    # 初始化 LLM 客户端
    try:
        print("🧠 初始化 LLM 客户端...")
        llm_client = HelloAgentsLLM()
        print(f"✅ LLM 客户端初始化成功，模型: {llm_client.model}")
    except Exception as e:
        print(f"❌ LLM 客户端初始化失败: {e}")
        print("\n📋 请创建 .env 文件并添加以下配置:")
        print("LLM_MODEL_ID=你的模型名称（如: qwen2.5:7b）")
        print("LLM_BASE_URL=http://localhost:11434/v1")
        print("LLM_API_KEY=ollama")
        print("LLM_TIMEOUT=60")
        return

    # 创建 Plan-and-Solve 智能体
    print("\n🤖 创建 Plan-and-Solve 智能体...")

    # 需要先定义 Planner 和 PlanAndSolveAgent 类
    # 这些类已经在文件顶部定义了，这里直接使用
    agent = PlanAndSolveAgent(llm_client)
    print("✅ 智能体创建成功")

    print(f"\n📝 测试问题:")
    print(f"   {test_question}")

    # 运行智能体
    print("\n" + "=" * 70)
    print("开始处理问题...")
    print("=" * 70)

    try:
        # 运行智能体处理问题
        answer = agent.run(test_question)

        print("\n" + "=" * 70)
        print(f"🎉 最终答案: {answer}")
        print("=" * 70)

    except KeyboardInterrupt:
        print("\n⚠️ 用户中断运行")
    except Exception as e:
        print(f"❌ 运行过程中出现错误: {e}")
        import traceback
        traceback.print_exc()


# 运行主函数
if __name__ == "__main__":
    main()

