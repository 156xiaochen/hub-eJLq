"""
智能对话系统信息解析 - 提示词测试脚本
基于 02-joint-bert-training-only 数据集的任务
"""

import json
from typing import Dict, Any

# 定义提示词模板
PROMPT_TEMPLATE = """你是一个智能对话语义解析助手。你的任务是分析用户输入，提取意图和实体信息。

## 可用意图列表（Intent）：
OPEN, SEARCH, REPLAY_ALL, NUMBER_QUERY, DIAL, CLOSEPRICE_QUERY, SEND, LAUNCH, 
PLAY, REPLY, RISERATE_QUERY, DOWNLOAD, QUERY, LOOK_BACK, CREATE, FORWARD, 
DATE_QUERY, SENDCONTACTS, DEFAULT, TRANSLATION, VIEW, ROUTE, POSITION

## 可用槽位类型（Slot）：
code, Src, startDate_dateOrig, film, endLoc_city, artistRole, location_country, 
location_area, author, startLoc_city, season, dishName, media, datetime_date, 
episode, teleOperator, questionWord, receiver, ingredient, name, startDate_time, 
startDate_date, location_province, endLoc_poi, artist, dynasty, area, location_poi, 
relIssue, Dest, content, keyword, target, startLoc_area, tvchannel, type, song, 
queryField, awayName, headNum, homeName, decade, payment, popularity, tag, 
startLoc_poi, date, startLoc_province, endLoc_province, location_city, absIssue, 
utensil, scoreDescr, dishName, endLoc_area, resolution, yesterday, timeDescr, 
category, subfocus, theatre, datetime_time

## 输出格式要求：
请严格按照以下 JSON 格式输出（不要输出其他任何内容）：
{{
    "intent": "意图名称",
    "slots": {{
        "槽位名 1": "槽位值 1",
        "槽位名 2": "槽位值 2"
    }}
}}

## 示例：
用户输入：请帮我打开 uc
输出：{{"intent": "LAUNCH", "slots": {{"name": "uc"}}}}

用户输入：查询许昌到中山的汽车票
输出：{{"intent": "QUERY", "slots": {{"Src": "许昌", "Dest": "中山"}}}}

用户输入：红烧肉怎么做
输出：{{"intent": "QUERY", "slots": {{"dishName": "红烧肉"}}}}

用户输入：把李会计的电话发给小江
输出：{{"intent": "SENDCONTACTS", "slots": {{"name": "李会计", "receiver": "小江"}}}}

## 现在开始处理：
用户输入：{user_input}
输出："""


class IntentParser:
    """意图解析器类"""
    
    def __init__(self):
        """
        初始化解析器
        
        Args:
            use_mock: 是否使用模拟模式（没有 API key 时使用规则匹配）
        """
        try:
            from openai import OpenAI
            self.client = OpenAI(
                api_key="sk-9f96f86d7029428bbb74d78d33859df2",
                base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"  # 阿里云通义千问 API 地址
            )
        except ImportError:
            print("未安装 openai 库，将使用模拟模式")
            self.use_mock = True
    
    def parse(self, user_input: str) -> Dict[str, Any]:
        """
        解析用户输入
        
        Args:
            user_input: 用户输入的文本
            
        Returns:
            包含 intent 和 slots 的字典
        """
        if self.use_mock:
            return self._mock_parse(user_input)
        else:
            return self._api_parse(user_input)
    
    def _mock_parse(self, user_input: str) -> Dict[str, Any]:
        """模拟解析（基于简单规则）"""
        import re
        
        user_input = user_input.lower()
        
        # App 打开
        if any(word in user_input for word in ['打开', '启动', '开', '启动']):
            app_names = ['微信', 'qq', 'uc', '淘宝', '百度', '地图', '音乐', '相机']
            for app in app_names:
                if app in user_input:
                    return {"intent": "LAUNCH", "slots": {"name": app}}
        
        # 菜谱查询
        if any(word in user_input for word in ['怎么做', '做法', '如何做', '菜谱']):
            # 提取菜名（简化版）
            dishes = ['红烧肉', '鱼香肉丝', '西红柿炒鸡蛋', '糖醋排骨', '酸菜鱼']
            for dish in dishes:
                if dish in user_input:
                    return {"intent": "QUERY", "slots": {"dishName": dish}}
        
        # 交通查询
        if any(word in user_input for word in ['到', '去', '车票', '高铁', '汽车']):
            cities = ['北京', '上海', '广州', '深圳', '许昌', '中山', '无锡', '阜阳']
            src = None
            dest = None
            for city in cities:
                if city in user_input:
                    if src is None:
                        src = city
                    else:
                        dest = city
            if src or dest:
                slots = {}
                if src:
                    slots['Src'] = src
                if dest:
                    slots['Dest'] = dest
                return {"intent": "QUERY", "slots": slots}
        
        # 联系人查询
        if any(word in user_input for word in ['电话', '号码', '联系方式']):
            if '发给' in user_input:
                return {"intent": "SENDCONTACTS", "slots": {"name": "未知", "receiver": "未知"}}
            else:
                return {"intent": "QUERY", "slots": {"name": "未知"}}
        
        # 默认返回
        return {"intent": "DEFAULT", "slots": {}}
    
    def _api_parse(self, user_input: str) -> Dict[str, Any]:
        """使用 API 解析"""
        prompt = PROMPT_TEMPLATE.format(user_input=user_input)
        
        response = self.client.chat.completions.create(
            model="qwen-turbo",  # 使用通义千问模型
            messages=[
                {"role": "system", "content": "你是一个专业的语义解析助手"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        result = response.choices[0].message.content.strip()
        
        # 提取 JSON 部分
        if "```json" in result:
            result = result.split("```json")[1].split("```")[0].strip()
        
        return json.loads(result)


def test_parser():
    """测试解析器"""
    parser = IntentParser()
    
    test_cases = [
        "请帮我打开微信",
        "查询许昌到中山的汽车票",
        "红烧肉怎么做",
        "把李会计的电话发给小江",
        "无锡到阜阳怎么坐汽车？",
        "打开 UC 浏览器",
        "鱼香肉丝的做法",
        "老王的电话号码是多少",
        "北京到上海的高铁",
        "播放周杰伦的稻香"
    ]
    
    print("=" * 60)
    print("智能对话系统信息解析测试")
    print("=" * 60)
    
    for i, text in enumerate(test_cases, 1):
        print(f"\n测试 {i}/{len(test_cases)}:")
        print(f"输入：{text}")
        
        result = parser.parse(text)
        
        print(f"输出：{json.dumps(result, ensure_ascii=False, indent=2)}")
        print("-" * 60)


if __name__ == "__main__":
    test_parser()
    
    print("\n\n使用说明：")
    print("1. 当前使用模拟模式（基于规则）")
    print("3. 可以使用 Qwen、ChatGPT、Kimi 等大模型 API")
