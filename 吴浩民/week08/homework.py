from pydantic import BaseModel, Field
from typing_extensions import Literal
import openai

client = openai.OpenAI(
    api_key="sk-92aa865c659e484a881281f153f521f2",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)

# 信息抽取智能体类, 用于从文本中提取结构化信息
class ExtractionAgent:
    def __init__(self, model_name: str):
        self.model_name = model_name

    # call方法负责把用户的输入变成结构化数据
    def call(self, user_prompt, response_model):
        messages = [
            {
                "role": "user",
                "content": user_prompt
            }
        ]
        # 根据response_model自动生成工具描述, 让模型知道要调用什么函数, 以及函数需要什么参数
        tools = [
            {
                "type": "function",
                "function": {
                    # 工具名字
                    "name": response_model.model_json_schema()['title'],
                    # 工具描述
                    "description": response_model.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        # 参数说明
                        "properties": response_model.model_json_schema()['properties'],
                        # 必须要传的参数
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
            # 提取的参数（json格式）
            arguments = response.choices[0].message.tool_calls[0].function.arguments

            # 参数转换为datamodel, 关注想要的参数
            return response_model.model_validate_json(arguments)
        except:
            print('ERROR', response.choices[0].message)
            return None

# 翻译信息抽取模型
class TranslationRequest(BaseModel):
    """从用户输入中提取翻译相关信息"""
    source_language: str = Field(description="原始语种")
    target_language: str = Field(description="目标语种")
    text: str = Field(description="待翻译的文本")

# 翻译智能体类
class TranslationAgent:
    def __init__(self, model_name: str):
        self.extraction_agent = ExtractionAgent(model_name)

    def extract_translation_info(self, user_input):
        # 提取翻译相关信息
        extraction_result = self.extraction_agent.call(user_input, TranslationRequest)
        if not extraction_result:
            return "无法解析翻译请求，请提供更明确的信息"
        
        # 构建结果，包含提取的信息
        result = {
            "原始语种": extraction_result.source_language,
            "目标语种": extraction_result.target_language,
            "待翻译的文本": extraction_result.text
        }
        
        return result

    def translate(self, user_input):
        # 提取翻译相关信息
        extraction_result = self.extraction_agent.call(user_input, TranslationRequest)
        if not extraction_result:
            return "无法解析翻译请求，请提供更明确的信息"
        
        # 构建翻译提示
        translation_prompt = f"请将以下{extraction_result.source_language}文本翻译成{extraction_result.target_language}：\n{extraction_result.text}"
        
        # 调用模型进行翻译
        messages = [
            {"role": "user", "content": translation_prompt}
        ]
        
        response = client.chat.completions.create(
            model="qwen-plus",
            messages=messages
        )
        
        # 构建结果，包含提取的信息和翻译结果
        result = {
            "原始语种": extraction_result.source_language,
            "目标语种": extraction_result.target_language,
            "待翻译的文本": extraction_result.text,
            "翻译结果": response.choices[0].message.content
        }
        
        return result

# 测试翻译智能体
if __name__ == "__main__":
    agent = TranslationAgent("qwen-plus")
    
    # 测试用例: 帮我将good！翻译为中文
    result = agent.extract_translation_info("帮我将good！翻译为中文")
    print("测试结果:")
    if isinstance(result, dict):
        for key, value in result.items():
            print(f"{key}: {value}")
    else:
        print(result)
