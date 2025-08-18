from json import tool

from transformers import pipeline
import torch

# # 检查并设置设备（Mac M芯片使用MPS）
# def get_device():
#     if torch.backends.mps.is_available():
#         return torch.device("mps")
#     elif torch.cuda.is_available():
#         return torch.device("cuda")
#     else:
#         return torch.device("cpu")
#
# device = get_device()
# print(f"使用设备: {device}")

# 明确指定模型名称和版本以避免警告，并指定设备
# classifier = pipeline(
#     "text-generation",
#     model="HuggingFaceTB/SmolLM2-360M",
#     device=torch.device("mps"),  # 指定使用的设备
#     max_length = 19,
#     num_return_sequences=2,
# )

# 执行情感分析并打印结果
# result = classifier("你 是 谁 ")
# print("restult:", result)


# classifier = pipeline("zero-shot-classification")
# result = classifier(
#     "This is a course about the Transformers library",
#     candidate_labels=["education", "politics", "business"],
#     device=torch.device("mps")
# )
# # 执行情感分析并打印结果
# print("restult:", result)


# from transformers import pipeline
#
# unmasker = pipeline("fill-mask")
# print("rs :",unmasker("who the  <mask> are you .", top_k=2))
#
# from smolagents import LiteLLMModel
#
# model = LiteLLMModel(
#     model_id="ollama_chat/qwen2:7b",  # Or try other Ollama-supported models
#     api_base="http://127.0.0.1:11434",  # Default Ollama local server
#     num_ctx=8192,
# )

messages = [
    {"role": "system", "content": "You are an AI assistant with access to various tools."},
    {"role": "user", "content": "Hi !"},
    {"role": "assistant", "content": "Hi human, what can help you with ?"},
]
# from transformers import AutoTokenizer
#
# tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM2-1.7B-Instruct")
# rendered_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
# print(rendered_prompt)

@tool
def calculator(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b

print(calculator.to_string())
