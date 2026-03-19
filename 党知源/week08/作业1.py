import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Type

import openai
from pydantic import BaseModel, Field


def _get_openai_client() -> openai.OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get(
        "OPENAI_BASE_URL",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    ).strip()

    if not api_key:
        raise RuntimeError(
            "未找到环境变量 OPENAI_API_KEY。请先设置后再运行。"
        )

    return openai.OpenAI(api_key=api_key, base_url=base_url)


class ExtractionAgent:

    def __init__(self, client: openai.OpenAI, model_name: str):
        self.client = client
        self.model_name = model_name

    def call(self, user_prompt: str, response_model: Type[BaseModel]) -> BaseModel:
        schema = response_model.model_json_schema()
        properties: Dict[str, Any] = schema.get("properties", {})
        required: List[str] = schema.get("required", [])

        tools = [
            {
                "type": "function",
                "function": {
                    "name": schema.get("title", "Extract"),
                    "description": schema.get("description", "Extract structured fields."),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        ]

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": user_prompt}],
            tools=tools,
            tool_choice="auto",
        )

        tool_calls = getattr(response.choices[0].message, "tool_calls", None)
        if not tool_calls:
            raise RuntimeError("模型未返回 tool_calls，无法进行结构化抽取。")

        arguments = tool_calls[0].function.arguments
        return response_model.model_validate_json(arguments)


class TranslationRequest(BaseModel):
    source_language: str = Field(
        description="原始语种，用中文描述（例如：英语、中文、日语、韩语等）。"
    )
    target_language: str = Field(
        description="目标语种，用中文描述（例如：英语、中文、日语、韩语等）。"
    )
    text: str = Field(
        description="待翻译的原始文本（不要包含“翻译为XXX/请翻译/类别”等额外指令）。"
    )


class TextTranslationAgent:
    def __init__(
        self,
        client: openai.OpenAI,
        extract_model: str = "qwen-plus",
        translate_model: str = "qwen-plus",
    ):
        self.client = client
        self.extract_agent = ExtractionAgent(client=client, model_name=extract_model)
        self.translate_model = translate_model

    def parse_request(self, user_text: str) -> TranslationRequest:
        req = self.extract_agent.call(user_text, TranslationRequest)
        return TranslationRequest.model_validate(req.model_dump())

    def translate(self, req: TranslationRequest) -> str:
        prompt = (
            f"请把下面文本从{req.source_language}翻译成{req.target_language}。"
            "只输出译文，不要输出任何解释、原文或额外标点说明。\n\n"
            f"文本：{req.text}"
        )

        completion = self.client.chat.completions.create(
            model=self.translate_model,
            messages=[{"role": "user", "content": prompt}],
            stream=False,
        )
        return completion.choices[0].message.content.strip()

    def run(self, user_text: str) -> Dict[str, str]:
        req = self.parse_request(user_text)
        translated = self.translate(req)

        return {
            "source_language": req.source_language,
            "target_language": req.target_language,
            "text": req.text,
            "translation": translated,
        }


if __name__ == "__main__":
    user_prompt = "good！翻译为中文"

    client = _get_openai_client()
    agent = TextTranslationAgent(
        client=client,
        extract_model="qwen-plus",
        translate_model="qwen-plus",
    )

    result = agent.run(user_prompt)
    print("原始语种:", result["source_language"])
    print("目标语种:", result["target_language"])
    print("待翻译的文本:", result["text"])
    print("翻译结果:", result["translation"])

