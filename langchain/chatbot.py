import getpass
import os
import sys
from langchain_core.messages import AIMessage
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from typing import Sequence

api = os.environ.get("OPEN_AI_TOKEN")
model = init_chat_model("gpt-5-mini", model_provider="openai"
                         , base_url="https://aihubmix.com/v1", api_key=api)


class State(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    language: str


prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability in {language}.",
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)




workflow = StateGraph(state_schema=State)
def call_model(state: State):
    prompt = prompt_template.invoke(state)
    response = model.invoke(prompt)
    return {"messages": response}



# Define the (single) node in the graph
workflow.add_edge(START, "model")
workflow.add_node("model", call_model)
app = workflow.compile(checkpointer=MemorySaver())

config = {"configurable": {"thread_id": "abc123"}}

def chat_loop():
    """启动命令行聊天循环"""
    print("🤖 聊天机器人已启动！")
    print("支持的命令：")
    print("  /language [lang] - 切换语言 (例如: /language Chinese)")
    print("  /quit - 退出聊天")
    print("  /help - 显示帮助")
    print()
    
    language = "Chinese"  # 默认中文
    print(f"当前语言: {language}")
    print("开始聊天吧！输入您的问题...")
    print("-" * 50)
    
    while True:
        try:
            user_input = input("\n👤 您: ").strip()
            
            if not user_input:
                continue
                
            # 处理命令
            if user_input.startswith("/"):
                if user_input == "/quit" or user_input == "/exit":
                    print("👋 再见！")
                    break
                elif user_input.startswith("/language"):
                    parts = user_input.split(maxsplit=1)
                    if len(parts) > 1:
                        language = parts[1]
                        print(f"🌐 已切换到 {language}")
                    else:
                        print("❌ 用法: /language [语言]")
                    continue
                elif user_input == "/help":
                    print("📋 可用命令:")
                    print("  /language [lang] - 切换语言")
                    print("  /quit, /exit - 退出")
                    print("  /help - 显示此帮助")
                    continue
                else:
                    print("❓ 未知命令，输入 /help 查看帮助")
                    continue
            
            # 处理正常对话
            input_messages = [HumanMessage(user_input)]
            output = app.invoke({"messages": input_messages, "language": language}, config)
            
            # 获取AI回复
            ai_response = output["messages"][-1].content
            print(f"🤖 AI: {ai_response}")
            
        except KeyboardInterrupt:
            print("\n👋 再见！")
            break
        except Exception as e:
            print(f"❌ 错误: {e}")

# 如果是直接运行此文件，启动聊天循环
if __name__ == "__main__":
    chat_loop()
else:
    # 保持向后兼容，如果是导入使用，执行原来的测试代码
    query1 = "Hi! I'm Bob. who are you"
    language = "Spanish"
    input_messages = [HumanMessage(query1)]
    output = app.invoke({"messages": input_messages, "language": language}, config)
    output["messages"][-1].pretty_print()  # output contains all messages in state
