---
name: stock-volatility-advisor
description: 复用 autostock 股票数据 skill，重点实现股票可视化功能：把股票日波动和周波动绘制在同一张图中，并结合 autostock 的股票分析方法给出简单买入、卖出或观望建议。
---

# 股票波动可视化与买卖建议

当用户希望把股票的日波动和周波动画在同一张图中，或希望基于波动大小获得简单买卖时机建议时，使用本 skill。

本 skill 的主任务是**股票波动可视化**：

- 获取股票日 K 和周 K 数据
- 分别计算日波动率和周波动率
- 将两条波动曲线绘制在同一张图中
- 基于图中的波动大小，结合 autostock 的股票分析方法，给出简单建议

本 skill 不重新定义股票分析方法，优先复用已有的 `skills/autostock/SKILL.md` 中的接口和分析规则。可用接口包括：

- `get_all_stock_code`：按代码或名称搜索股票
- `get_all_index_code`：查询指数代码
- `get_stock_industry_code`：获取行业/板块数据
- `get_board_info`：获取大盘数据
- `get_stock_rank`：获取股票排行
- `get_month_line`：获取月 K 线数据
- `get_stock_minute_data`：获取分时数据
- `get_day_line`：获取日 K 线数据
- `get_week_line`：获取周 K 线数据
- `get_stock_info`：获取股票基础信息

## 工作流程

1. 获取或确认股票代码；如果用户输入名称，先用 `get_all_stock_code` 查询代码。
2. 调用 `get_day_line` 和 `get_week_line` 获取日 K、周 K 数据。
3. 对日 K 和周 K 计算波动率：
   - `volatility = (high - low) / close * 100`
4. 在同一张图中绘制：
   - 日波动曲线
   - 周波动曲线
5. 为了让建议更完整，可以补充调用 `get_stock_info`、`get_board_info`、`get_stock_industry_code`、`get_stock_rank`、`get_month_line`、`get_stock_minute_data`。
6. 分析判断必须优先遵循 `autostock/SKILL.md` 中 `# 股票分析方法` 部分：
   - 月 K 判断大趋势：上升、下降或震荡。
   - 周 K 判断支撑、压力和中期节奏。
   - 日 K 判断短期买入、卖出或观望信号。
   - 波动率图只作为辅助判断：日波动放大代表短线风险上升，日波动收敛且价格走强代表可能出现试探买点。

## 输出内容

脚本输出：

- 图表文件路径
- 股票基础信息摘要
- 大盘、板块和排行背景摘要
- 近期日波动率
- 近期周波动率
- 月 K / 周 K / 日 K 的简要趋势判断
- 最新收盘价
- 操作建议
- 简短理由
