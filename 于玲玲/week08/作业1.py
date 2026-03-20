from pydantic import BaseModel, Field
from typing import Optional
import openai

client = openai.OpenAI(
    api_key="sk-f0ab3fca58044adcb75b5a60974549b3", # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

class TranslationResult(BaseModel):
    """翻译结果结构"""
    translated_text: str = Field(description="翻译后的文本内容")
    source_lang: Optional[str] = Field(default=None, description="原始语种，如 'zh', 'en', 'ja'")
    target_lang: str = Field(description="目标语种，如 'en', 'zh', 'fr'")

class TranslationAgent:
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name

    def translate(self, user_prompt: str, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        # 传入需要提取的内容，自己写了一个tool格式
        tools = [
            {
                "type": "function",
                "function": {
                    "name": response_model.model_json_schema()['title'],  # 工具名字
                    "description": response_model.model_json_schema()['description'],  # 工具描述
                    "parameters": {
                        "type": "object",
                        "properties": response_model.model_json_schema()['properties'],  # 参数说明
                        "required": response_model.model_json_schema()['required'],  # 必须要传的参数
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
            # 提取的参数（json格式）
            arguments = response.choices[0].message.tool_calls[0].function.arguments

            # 参数转换为datamodel，关注想要的参数
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None



if __name__ == "__main__":
    agent = TranslationAgent(model_name="qwen-plus")

    # 示例 1: 中译英
    result1 = agent.translate("今天天气真好，适合出去散步。", TranslationResult)
    print("中译英:", result1.translated_text )

    # 示例 2: 英译中
    result2 = agent.translate("Artificial intelligence is transforming the world.", TranslationResult)
    print("英译中:", result2.translated_text )

    # 示例 3: 日译中
    result3 = agent.translate("こんにちは、元気ですか？", TranslationResult)
    print("日译中:", result3.translated_text )