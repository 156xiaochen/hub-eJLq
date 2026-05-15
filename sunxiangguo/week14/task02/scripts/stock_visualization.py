"""
股票可视化分析脚本
功能：绘制股票的周波动和日波动曲线，并给出买卖建议
"""

import os
import warnings
warnings.filterwarnings('ignore')

# 必须在导入pyplot之前设置后端
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# AutoStock API配置
AUTOSTOCK_TOKEN = "zgaLG8unUPr"
BASE_URL = "https://api.autostock.cn/v1"

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False


class StockVisualizer:
    """股票可视化分析器"""
    
    def __init__(self, stock_code, start_date, end_date):
        """
        初始化股票分析器
        
        参数:
            stock_code: 股票代码 (如: 600519.SH)
            start_date: 开始日期 (格式: YYYY-MM-DD)
            end_date: 结束日期 (格式: YYYY-MM-DD)
        """
        self.stock_code = stock_code
        self.start_date = start_date
        self.end_date = end_date
        self.df = None
        self.weekly_df = None
        
    def fetch_data(self):
        """获取股票数据"""
        print(f"正在获取 {self.stock_code} 的数据...")
        
        # 转换股票代码格式：600519.SH -> sh600519
        code = self.stock_code.lower().replace('.sh', '').replace('.sz', '')
        if 'sh' in self.stock_code.lower():
            code = f"sh{code}"
        elif 'sz' in self.stock_code.lower():
            code = f"sz{code}"
        
        # 获取日K线数据
        url = f"{BASE_URL}/stock/kline/day?token={AUTOSTOCK_TOKEN}"
        params = {
            "code": code,
            "startDate": self.start_date,
            "endDate": self.end_date,
            "type": 1  # 前复权
        }
        
        try:
            response = requests.get(url, params=params, timeout=10)
            data = response.json()
            
            if data.get("code") != 200 or not data.get("data"):
                raise Exception(f"API返回错误: {data.get('message', '未知错误')}")
            
            # 解析数据: [["日期", "昨收", "今开", "最高", "最低", "成交量"], ...]
            raw_data = data["data"]
            df = pd.DataFrame(raw_data, columns=['date', 'close_prev', 'open', 'high', 'low', 'volume'])
            
            # 数据类型转换
            df['date'] = pd.to_datetime(df['date'])
            for col in ['open', 'high', 'low', 'close_prev', 'volume']:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # 重命名
            df.rename(columns={'close_prev': 'close'}, inplace=True)
            
            # 计算收盘价（使用昨收作为参考）
            if 'close' not in df.columns or df['close'].isna().all():
                df['close'] = df['open']  # 如果没有收盘价，用开盘价替代
            
            self.df = df
            
            # 计算周线数据
            df['year_week'] = df['date'].dt.strftime('%Y-W%U')
            weekly_df = df.groupby('year_week').agg({
                'date': 'last',
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).reset_index(drop=True)
            
            self.weekly_df = weekly_df
            print(f"✓ 数据获取完成: {len(df)} 个交易日, {len(weekly_df)} 个交易周")
            
        except Exception as e:
            print(f"✗ 数据获取失败: {e}")
            raise
        
    def calculate_metrics(self):
        """计算技术指标"""
        # 日收益率和波动率
        self.df['daily_return'] = self.df['close'].pct_change()
        self.df['daily_volatility'] = self.df['daily_return'].rolling(window=5).std()
        
        # 周收益率和波动率
        self.weekly_df['weekly_return'] = self.weekly_df['close'].pct_change()
        self.weekly_df['weekly_volatility'] = self.weekly_df['weekly_return'].rolling(window=4).std()
        
        # 移动平均线
        self.df['MA5'] = self.df['close'].rolling(window=5).mean()
        self.df['MA20'] = self.df['close'].rolling(window=20).mean()
        self.df['MA60'] = self.df['close'].rolling(window=60).mean()
        
    def plot_chart(self, save_path='stock_analysis.png'):
        """绘制股票分析图表"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(4, 1, height_ratios=[4, 1, 1, 1], hspace=0.3)
        
        # 主图：价格走势（日线 + 周线）
        ax1 = fig.add_subplot(gs[0])
        ax1.plot(self.df['date'], self.df['close'], label='日线收盘价', 
                linewidth=1, color='steelblue', alpha=0.7)
        ax1.plot(self.weekly_df['date'], self.weekly_df['close'], 
                label='周线收盘价', linewidth=2.5, color='red', marker='o', 
                markersize=5, markerfacecolor='darkred')
        
        # 添加均线
        ax1.plot(self.df['date'], self.df['MA20'], label='MA20', 
                linewidth=1.5, linestyle='--', color='orange', alpha=0.8)
        ax1.plot(self.df['date'], self.df['MA60'], label='MA60', 
                linewidth=1.5, linestyle='--', color='green', alpha=0.8)
        
        ax1.set_title(f'{self.stock_code} 股票价格走势分析\n{self.start_date} 至 {self.end_date}', 
                     fontsize=14, fontweight='bold')
        ax1.legend(loc='best', fontsize=10)
        ax1.grid(True, alpha=0.3)
        ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        ax1.xaxis.set_major_locator(mdates.MonthLocator())
        plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45)
        
        # 副图1：日波动率
        ax2 = fig.add_subplot(gs[1])
        ax2.fill_between(self.df['date'], self.df['daily_volatility'], 
                        alpha=0.5, color='purple', label='日波动率(5日)')
        ax2.axhline(y=self.df['daily_volatility'].mean(), color='red', 
                   linestyle='--', label=f'平均波动率: {self.df["daily_volatility"].mean():.2%}')
        ax2.set_ylabel('波动率', fontsize=10)
        ax2.legend(loc='best', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        # 副图2：成交量
        ax3 = fig.add_subplot(gs[2])
        colors = ['red' if self.df.iloc[i]['close'] >= self.df.iloc[i]['open'] 
                 else 'green' for i in range(len(self.df))]
        ax3.bar(self.df['date'], self.df['volume'], color=colors, alpha=0.6, label='成交量')
        ax3.set_ylabel('成交量', fontsize=10)
        ax3.legend(loc='best', fontsize=9)
        ax3.grid(True, alpha=0.3)
        ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45)
        
        # 副图3：日收益率
        ax4 = fig.add_subplot(gs[3])
        colors_ret = ['red' if x >= 0 else 'green' for x in self.df['daily_return']]
        ax4.bar(self.df['date'], self.df['daily_return'] * 100, 
               color=colors_ret, alpha=0.6, label='日收益率')
        ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax4.set_ylabel('收益率(%)', fontsize=10)
        ax4.legend(loc='best', fontsize=9)
        ax4.grid(True, alpha=0.3)
        ax4.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax4.xaxis.get_majorticklabels(), rotation=45)
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"✓ 图表已保存至: {save_path}")
        # 注释掉plt.show()，因为在非交互式环境中不需要显示
        # plt.show()
        
    def generate_recommendation(self):
        """生成交易建议"""
        # 计算统计数据
        daily_vol_mean = self.df['daily_volatility'].mean()
        weekly_vol_mean = self.weekly_df['weekly_volatility'].mean()
        
        recent_price = self.df.tail(5)['close'].mean()
        earlier_price = self.df.head(5)['close'].mean()
        trend = "上涨" if recent_price > earlier_price else "下跌"
        
        # 找出关键高低点
        max_price = self.df['high'].max()
        min_price = self.df['low'].min()
        current_price = self.df['close'].iloc[-1]
        
        # 支撑阻力位
        resistance = max_price * 0.98
        support = min_price * 1.02
        
        recommendation = f"""
{'='*60}
【股票分析报告】{self.stock_code}
分析周期: {self.start_date} 至 {self.end_date}
{'='*60}

📊 基本统计:
  - 当前价格: ¥{current_price:.2f}
  - 期间最高: ¥{max_price:.2f}
  - 期间最低: ¥{min_price:.2f}
  - 整体趋势: {trend}
  
📈 波动性分析:
  - 日均波动率: {daily_vol_mean:.2%}
  - 周均波动率: {weekly_vol_mean:.2%}
  - 波动特征: {'高波动' if daily_vol_mean > 0.03 else '中等波动' if daily_vol_mean > 0.015 else '低波动'}

💡 关键技术位:
  - 阻力位: ¥{resistance:.2f}
  - 支撑位: ¥{support:.2f}
  - 当前位置: {'接近阻力位' if current_price > resistance * 0.95 else '接近支撑位' if current_price < support * 1.05 else '中间位置'}

🎯 交易建议:

【买入时机】
  1. 当价格回调至支撑位附近 ({support:.2f}) 且波动率降低时
  2. 当日线突破MA20均线且成交量放大时
  3. 当周线显示连续3周以上企稳迹象时
  4. 建议在波动率低于平均值 {daily_vol_mean:.2%} 时建仓

【卖出时机】
  1. 当价格接近阻力位 ({resistance:.2f}) 且出现滞涨信号时
  2. 当日线跌破MA20均线且成交量萎缩时
  3. 当单日波动率超过平均值的2倍时考虑减仓
  4. 达到预期收益目标（建议15-20%）时分批止盈

⚠️ 风险控制:
  - 止损位: 建议设置在建仓价下方 5-8%
  - 仓位管理: 单只股票不超过总资金的 20%
  - 分批建仓: 首次建仓 50%，确认趋势后加仓
  - 关注大盘: 结合市场整体走势判断

📌 注意事项:
  - 本分析基于历史数据，不构成投资建议
  - 请结合基本面分析和市场消息综合判断
  - 高波动时期应降低仓位，控制风险
  - 定期复盘，根据市场变化调整策略

{'='*60}
        """
        
        print(recommendation)
        return recommendation


def analyze_stock(stock_code="600519.SH", start_date="2024-01-01", end_date="2024-12-31"):
    """
    执行完整的股票分析流程
    
    参数:
        stock_code: 股票代码
        start_date: 开始日期
        end_date: 结束日期
    
    返回:
        交易建议文本
    """
    # 创建分析器
    analyzer = StockVisualizer(stock_code, start_date, end_date)
    
    # 获取数据
    analyzer.fetch_data()
    
    # 计算指标
    analyzer.calculate_metrics()
    
    # 绘制图表
    output_file = f'stock_{stock_code.replace(".", "_")}_analysis.png'
    analyzer.plot_chart(output_file)
    
    # 生成建议
    recommendation = analyzer.generate_recommendation()
    
    return recommendation


if __name__ == "__main__":
    import sys
    
    # 从命令行参数获取股票代码和日期范围
    if len(sys.argv) >= 2:
        stock_code = sys.argv[1]
        start_date = sys.argv[2] if len(sys.argv) >= 3 else "2024-01-01"
        end_date = sys.argv[3] if len(sys.argv) >= 4 else "2024-12-31"
    else:
        # 默认分析贵州茅台
        stock_code = "600519.SH"
        start_date = "2026-01-01"
        end_date = "2026-05-14"
    
    print(f"\n开始分析股票: {stock_code}")
    print(f"时间范围: {start_date} 至 {end_date}\n")
    
    analyze_stock(stock_code, start_date, end_date)
