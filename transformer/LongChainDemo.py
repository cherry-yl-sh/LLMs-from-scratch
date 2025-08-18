import getpass
import os
from langchain.chat_models import init_chat_model
api = os.environ.get("OPEN_AI_TOKEN")
model = init_chat_model("gpt-4o-mini", model_provider="openai"
,base_url ="https://aihubmix.com/v1",api_key=api )

print(model.invoke("Hello, world!")
      )
