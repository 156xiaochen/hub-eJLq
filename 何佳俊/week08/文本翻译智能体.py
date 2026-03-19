from typing import Literal

import openai
from pydantic import BaseModel, Field  # 定义传入的数据请求格式

# 初始化 OpenAI 客户端（使用阿里云百炼平台）
client = openai.OpenAI(
    api_key="sk-9f96f86d7029428bbb74d78d33859df22",  # https://bailian.console.aliyun.com/?tab=model#/api-key
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
)


class TranslationRequest(BaseModel):
    """文本翻译请求信息抽取"""
    source_language: Literal["中文", "英文", "日文", "韩文", "法文", "德文", "西班牙文", "俄文", "其他"] = Field(
        description="原始语种，待翻译文本的语言"
    )
    target_language: Literal["中文", "英文", "日文", "韩文", "法文", "德文", "西班牙文", "俄文", "其他"] = Field(
        description="目标语种，要翻译成的语言"
    )
    text_to_translate: str = Field(
        description="待翻译的文本内容"
    )


class TranslationAgent:
    """文本翻译智能体"""
    
    def __init__(self, model_name: str = "qwen-plus"):
        self.model_name = model_name
    
    def extract_translation_info(self, user_input: str) -> TranslationRequest:
        """
        从用户输入中提取翻译请求信息
        
        Args:
            user_input: 用户的自然语言输入
            
        Returns:
            TranslationRequest: 包含原始语种、目标语种和待翻译文本的对象
        """
        messages = [
            {
                "role": "system",
                "content": "你是一个文本翻译助手，能够从用户的自然语言输入中准确提取翻译请求的关键信息。"
            },
            {
                "role": "user",
                "content": user_input
            }
        ]
        
        # 构建 tools 定义
        tools = [
            {
                "type": "function",
                "function": {
                    "name": TranslationRequest.model_json_schema()['title'],
                    "description": TranslationRequest.model_json_schema()['description'],
                    "parameters": {
                        "type": "object",
                        "properties": TranslationRequest.model_json_schema()['properties'],
                        "required": TranslationRequest.model_json_schema()['required'],
                    },
                }
            }
        ]
        
        # 调用大模型
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            tools=tools,
            tool_choice="auto",
        )
        
        try:
            arguments = response.choices[0].message.tool_calls[0].function.arguments
            return TranslationRequest.model_validate_json(arguments)
        except Exception as e:
            print(f'解析错误：{e}')
            print(f'模型响应：{response.choices[0].message}')
            return None
    
    def translate(self, translation_request: TranslationRequest) -> str:
        """
        执行实际的翻译任务
        
        Args:
            translation_request: 翻译请求对象
            
        Returns:
            str: 翻译结果
        """
        messages = [
            {
                "role": "system",
                "content": f"你是一个专业的翻译助手，擅长将文本从{translation_request.source_language}翻译到{translation_request.target_language}。请保持原文的语气和风格。"
            },
            {
                "role": "user",
                "content": f"请将以下{translation_request.source_language}文本翻译成{translation_request.target_language}：\n\n{translation_request.text_to_translate}"
            }
        ]
        
        response = client.chat.completions.create(
            model=self.model_name,
            messages=messages,
        )
        
        return response.choices[0].message.content
    
    def process(self, user_input: str) -> dict:
        """
        完整的翻译流程：提取信息 + 执行翻译
        
        Args:
            user_input: 用户的自然语言输入
            
        Returns:
            dict: 包含翻译请求信息和翻译结果的字典
        """
        # 步骤 1：提取翻译请求信息
        print("=" * 60)
        print("步骤 1: 分析用户需求，提取翻译信息")
        print("=" * 60)
        
        translation_request = self.extract_translation_info(user_input)
        
        if not translation_request:
            return {
                "success": False,
                "error": "无法解析用户的翻译请求",
                "original_input": user_input
            }
        
        print(f"✓ 原始语种：{translation_request.source_language}")
        print(f"✓ 目标语种：{translation_request.target_language}")
        print(f"✓ 待翻译文本：{translation_request.text_to_translate}")
        print()
        
        # 步骤 2：执行翻译
        print("=" * 60)
        print("步骤 2: 执行翻译")
        print("=" * 60)
        
        translation_result = self.translate(translation_request)
        
        print(f"✓ 翻译结果：{translation_result}")
        print()
        
        return {
            "success": True,
            "request": translation_request,
            "result": translation_result
        }


def main():
    """主函数 - 测试翻译智能体"""
    agent = TranslationAgent(model_name="qwen-plus")
    
    # 测试用例 1：明确的翻译请求
    print("\n" + "=" * 60)
    print("测试用例 1: 明确的翻译请求")
    print("=" * 60)
    user_input_1 = "帮我将 good！翻译为中文"
    print(f"用户输入：{user_input_1}")
    result_1 = agent.process(user_input_1)
    
    if result_1["success"]:
        print(f"\n【翻译结果】")
        print(f"原文 ({result_1['request'].source_language}): {result_1['request'].text_to_translate}")
        print(f"译文 ({result_1['request'].target_language}): {result_1['result']}")
    
    # 测试用例 2：带标点的短句
    print("\n" + "=" * 60)
    print("测试用例 2: 带标点的短句")
    print("=" * 60)
    user_input_2 = "把'Hello, how are you?'翻译成中文"
    print(f"用户输入：{user_input_2}")
    result_2 = agent.process(user_input_2)
    
    if result_2["success"]:
        print(f"\n【翻译结果】")
        print(f"原文 ({result_2['request'].source_language}): {result_2['request'].text_to_translate}")
        print(f"译文 ({result_2['request'].target_language}): {result_2['result']}")
    
    # 测试用例 3：长文本翻译
    print("\n" + "=" * 60)
    print("测试用例 3: 长文本翻译")
    print("=" * 60)
    user_input_3 = "Please translate the following Chinese to English: 今天天气很好，我们一起去公园散步吧。"
    print(f"用户输入：{user_input_3}")
    result_3 = agent.process(user_input_3)
    
    if result_3["success"]:
        print(f"\n【翻译结果】")
        print(f"原文 ({result_3['request'].source_language}): {result_3['request'].text_to_translate}")
        print(f"译文 ({result_3['request'].target_language}): {result_3['result']}")


if __name__ == "__main__":
    main()
