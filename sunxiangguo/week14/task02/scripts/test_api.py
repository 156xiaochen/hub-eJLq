"""
测试AutoStock API连接
"""

import requests

# AutoStock API配置
AUTOSTOCK_TOKEN = "zgaLG8unUPr"
BASE_URL = "https://api.autostock.cn/v1"

def test_api():
    """测试API连接"""
    
    # 测试1: 获取日K线数据（贵州茅台）
    print("="*60)
    print("测试1: 获取贵州茅台(600519.SH)日K线数据")
    print("="*60)
    
    code = "sh600519"
    url = f"{BASE_URL}/stock/kline/day?token={AUTOSTOCK_TOKEN}"
    params = {
        "code": code,
        "startDate": "2024-01-01",
        "endDate": "2024-12-31",
        "type": 1
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        print(f"响应状态码: {data.get('code')}")
        print(f"响应消息: {data.get('message')}")
        
        if data.get("code") == 200 and data.get("data"):
            print(f"✓ 成功获取数据，共 {len(data['data'])} 条记录")
            print(f"\n前3条数据示例:")
            for i, record in enumerate(data['data'][:3], 1):
                print(f"  {i}. {record}")
            return True
        else:
            print(f"✗ 获取数据失败: {data.get('message', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return False


def test_stock_info():
    """测试获取股票基本信息"""
    print("\n" + "="*60)
    print("测试2: 获取贵州茅台(600519.SH)基本信息")
    print("="*60)
    
    code = "sh600519"
    url = f"{BASE_URL}/stock?token={AUTOSTOCK_TOKEN}&code={code}"
    
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        
        print(f"响应状态码: {data.get('code')}")
        
        if data.get("code") == 200 and data.get("data"):
            print(f"✓ 成功获取股票信息")
            stock_info = data['data']
            print(f"  股票名称: {stock_info.get('name')}")
            print(f"  股票代码: {stock_info.get('code')}")
            print(f"  当前价格: {stock_info.get('price')}")
            return True
        else:
            print(f"✗ 获取信息失败: {data.get('message', '未知错误')}")
            return False
            
    except Exception as e:
        print(f"✗ 请求失败: {e}")
        return False


if __name__ == "__main__":
    print("\n开始测试AutoStock API...\n")
    
    result1 = test_api()
    result2 = test_stock_info()
    
    print("\n" + "="*60)
    print("测试结果总结")
    print("="*60)
    print(f"日K线数据获取: {'✓ 通过' if result1 else '✗ 失败'}")
    print(f"股票信息获取: {'✓ 通过' if result2 else '✗ 失败'}")
    
    if result1 and result2:
        print("\n🎉 所有测试通过！API工作正常。")
    else:
        print("\n⚠️  部分测试失败，请检查网络连接或API Token。")
