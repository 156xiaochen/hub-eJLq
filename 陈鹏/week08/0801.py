from pydantic import BaseModel, Field  # 定义传入的数据请求格式
from typing import List
from typing_extensions import Literal

import openai
import json

client = openai.OpenAI(
    api_key="sk-f0ab3fca58044adcb75b5a60974549b3",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 本质是function call
# 可以传入多个待选函数，让大模型选择其中一个
# 传的是我们的函数的描述，让大模型选择，生成调用这个函数的传入参数
tools = [
    {
        "type": "function",
        "function": {
            "name": "Translate",
            "description": "文本翻译智能体",
            "parameters": {
                "type": "object",
                "properties": {
                    "sourceLanguage": {
                        "description": "原始语种",
                        "title": "SourceLanguage",
                        "type": "string",
                    },
                    "targetLanguage": {
                        "description": "目标语种",
                        "title": "TargetLanguage",
                        "type": "string",
                    },
                    "sourceText": {
                        "description": "待翻译的文本",
                        "title": "SourceText",
                        "type": "string",
                    },
                    "targetText": {
                        "description": "翻译后的文本",
                        "title": "TargetText",
                        "type": "string",
                    },
                },
                "required": ["sourceLanguage", "targetLanguage", "sourceText"],
            },
        },
    }
]

messages = [
    {
        "role": "user",
        "content": "帮我将good！翻译为中文？"
    }
]

# 大模型选择了一个函数，生成了函数的调用过程， 这也是agent 的核心功能
response = client.chat.completions.create(
    model="qwen-plus",
    messages=messages,
    tools=tools,  # 生成函数的调用方式，并不是所有的模型都支持（某些比较小的模型不支持）
    tool_choice="auto",
)
print(response.choices[0].message.tool_calls[0].function)

"""
这个智能体（不是满足agent所有的功能），能自动生成tools的json，实现信息信息抽取
指定写的tool的格式
"""


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
            tool_choice="auto",  # 自动选择工具
        )
        try:
            # 提取的参数（json格式）
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            # 解析抽取的参数为字典
            args_dict = json.loads(arguments)

            # 构造翻译提示词
            translate_prompt = f"将{args_dict['sourceLanguage']}的「{args_dict['sourceText']}」翻译为{args_dict['targetLanguage']}，仅返回翻译结果"
            # 调用大模型生成翻译结果
            translate_resp = client.chat.completions.create(
                model=self.model_name,
                messages=[{"role": "user", "content": translate_prompt}],
                temperature=0.1
            )
            # 填充翻译结果到参数字典
            args_dict['targetText'] = translate_resp.choices[0].message.content.strip()
            # 重新转为JSON字符串用于验证模型
            arguments = json.dumps(args_dict)
            # ========== 新增结束 ==========

            # 参数转换为datamodel，关注想要的参数
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None


class TranslateRequest(BaseModel):
    """自动翻译"""
    sourceLanguage: str = Field(description="原始语种，例如：中文、英文")
    targetLanguage: str = Field(description="目标语种，例如：中文、韩文")
    sourceText: str = Field(description="待翻译的文本内容")
    targetText: str = Field(description="翻译后的文本内容")

if __name__ == "__main__":
    result = ExtractionAgent(model_name="qwen-plus").call("帮我将good！翻译为韩文", TranslateRequest)
    print(result)

    result = ExtractionAgent(model_name="qwen-plus").call("请把“我今天很开心”翻译成英文", TranslateRequest)
    print(result)
