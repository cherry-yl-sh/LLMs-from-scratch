import os
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from typing import Sequence
from langchain_core.messages import trim_messages, AIMessage

api = os.environ.get("OPEN_AI_TOKEN")
model = init_chat_model("gpt-5-mini", model_provider="openai"
                        , base_url="https://aihubmix.com/v1", api_key=api)
full_response = ""
for chunk in model.stream("who are you"):
    if chunk.content:
        print(chunk.content, end="", flush=True)
        full_response += chunk.content