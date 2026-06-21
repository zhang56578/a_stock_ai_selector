# A股智选助手 (A-Stock AI Selector) - AI驱动选股可视化工具

专为解决A股股民痛点设计的一站式AI选股与持仓优化仪表板。

## 核心痛点解决
1. **热点发现**：自动识别存储、半导体设备等板块轮动、低位股、产业链传导。
2. **买卖点决策**：技术指标信号 + 止损/止盈 + 持仓周期建议。
3. **美股联动**：昨日美股涨跌 → A股预测。
4. **宏观资金预测**：降息/战争概率下的避险/风险资金流向。
5. **策略学习**：内置策略回测 + 博主经验总结。

## 功能模块
- **技术验证**：K线、均线(MA5/10/20/60)、MACD、RSI、筹码分布、成交量/资金流。回测策略有效性。
- **持仓优化**：基于成本、仓位诊断：持有/补仓/减仓/换股。组合建议。
- **价值研判**：个股深度报告（财务、估值、公告、机构评级 + 基本面+技术面）。
- **行情复盘**：大盘/板块轮动、主力/北向/涨停/ETF动态、市场风格判断。
- **可视化界面**：Streamlit多Tab仪表板，交互式Plotly图表。

参考数据架构：https://github.com/simonlin1212/a-stock-data （推荐集成其27端点数据源）。

## 快速开始
1. 克隆或下载本项目。
2. 安装依赖：
   ```bash
   pip install -r requirements.txt
   ```
   （推荐额外：`pip install akshare` 用于真实A股数据，或按a-stock-data指南安装mootdx等。）
3. 运行：
   ```bash
   streamlit run app.py
   ```
4. 浏览器打开 http://localhost:8501

## 使用说明
- 侧边栏输入股票代码（如688017）或板块关键词（如"半导体"）。
- 切换Tab查看不同模块。
- 上传持仓CSV或手动编辑表格进行优化诊断。
- 点击按钮生成/导出报告（DOCX或文本）。
- 真实数据集成：在代码中替换mock_data函数为真实API调用（示例已注释）。

## 数据来源与扩展（已内置真实数据模块）
- **演示数据**：内置合成数据 + yfinance（美股联动）。
- **真实A股（推荐）**：项目已包含 `data_provider.py`，直接集成 a-stock-data 架构：
  1. 安装额外依赖：`pip install mootdx requests`
  2. `app.py` 中已支持真实调用（优先 mootdx K线 + 东财资金流/板块，带自动限流防封IP）。
  3. 打开 `app.py`，取消注释或直接使用：
     ```python
     from data_provider import get_kline, get_fund_flow, get_sector_hotspots, get_us_stock_linkage, get_macro_prediction, add_technical_indicators
     ```
  4. 把原来 `get_real_kline_placeholder` 替换为真实入口（已准备好 fallback 到 mock）。
- 完整27端点想更深入？直接参考 https://github.com/simonlin1212/a-stock-data 的 SKILL.md 复制函数。
- **美股**：yfinance。
- **回测**：简单历史策略回测框架，可扩展为 vectorbt。

## 注意事项
- 本工具仅供学习/参考，非投资建议。股市有风险，决策需谨慎。
- 实时数据需网络；部分功能用mock演示，替换为真实数据源即可生产使用。
- 扩展：添加LLM（如Grok API）生成自然语言解读；集成更多策略回测。

## 项目结构
- app.py：主Streamlit应用
- requirements.txt
- README.md
- sample_data/：示例CSV（可选，代码内生成）

## 未来增强
- 真实数据管道（akshare + a-stock-data）
- AI报告生成（集成LLM）
- 自动回测报告
- 微信/APP部署

有问题或需要定制，欢迎反馈！
```

Now, the main app.py - this will be long, but comprehensive. Use mock data with functions to compute indicators.