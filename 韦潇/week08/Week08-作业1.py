from pydantic import BaseModel, Field
from typing_extensions import Literal
import openai

client = openai.OpenAI(
    api_key="sk-f0ab3fca58044adcb75b5a60974549b3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],
                        "required": response_model.model_json_schema()['required'],
                    },
                }
            }
        ]

        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


class TranslationRequest(BaseModel):
    """解析翻译请求，识别原始语种、目标语种和待翻译文本"""
    source_language: str = Field(description="原始语种，即待翻译文本的语言")
    target_language: str = Field(description="目标语种，即要翻译成的语言")
    text_to_translate: str = Field(description="待翻译的文本内容")


if __name__ == "__main__":
    agent = ExtractionAgent(model_name="qwen-plus")

    test_cases = [
        "帮我将good！翻译为中文",
        "请把'Hello, how are you?'翻译成法语",
        "把这句话'今天天气真好'翻译成英语",
        "将'Je t'aime'翻译为日语",
        "请将'人工智能正在改变世界'翻译成德语",
    ]

    print("=" * 60)
    print("文本翻译智能体 - 自动识别翻译信息")
    print("=" * 60)

    for user_input in test_cases:
        print(f"\n用户输入: {user_input}")
        print("-" * 40)
        result = agent.call(user_input, TranslationRequest)
        if result:
            print(f"原始语种: {result.source_language}")
            print(f"目标语种: {result.target_language}")
            print(f"待翻译文本: {result.text_to_translate}")
        print("-" * 40)
