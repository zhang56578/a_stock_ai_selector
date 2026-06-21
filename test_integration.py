#!/usr/bin/env python3
"""集成测试 - 验证所有模块协作正常"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from technical_indicators import TechnicalIndicators, add_indicators_streamlined
from ai_engine import AIEngine
from backtest_engine import run_backtest, compare_strategies
import hashlib

np.random.seed(42)
dates = pd.date_range(end=datetime.now(), periods=150, freq='B')
trend = np.cumsum(np.random.normal(0.0002, 0.015, 150))
prices = 100 * np.exp(trend)
data = []
for i, date in enumerate(dates):
    base = prices[i]
    vol = np.random.uniform(0.01, 0.025)
    data.append({
        'date': date,
        'open': round(base * (1 + np.random.normal(0, vol * 0.5)), 2),
        'high': round(base * (1 + abs(np.random.normal(0, vol))), 2),
        'low': round(base * (1 - abs(np.random.normal(0, vol))), 2),
        'close': round(base, 2),
        'volume': int(np.random.uniform(5e6, 5e7)),
    })
df = pd.DataFrame(data).set_index('date')
df = add_indicators_streamlined(df)
print(f"Mock data: {len(df)} rows, {len(df.columns)} cols")

# Test AI engine
engine = AIEngine()
signals = engine.detect_signals(df)
risk = engine.calculate_risk_score(df, signals)
levels = engine.calculate_trade_levels(df)
chip = TechnicalIndicators(df.copy()).get_chip_distribution()
fund_flow = {'主力净流入(亿)': 5.2, '大单净流入(亿)': 3.1, '北向资金(亿)': 1.5, '趋势': '净流入'}
report = engine.generate_report('688017', '半导体龙头', '半导体设备', df, signals, risk, levels, fund_flow, chip)

print(f"AI Signals: {signals['overall']} (B:{signals['buy_score']} S:{signals['sell_score']})")
print(f"Risk: {risk['level']} ({risk['score']}/100)")
print(f"Report generated: {len(report)} chars")
print(f"Levels: entry={levels['buy_point_aggressive']}, stop={levels['stop_loss_wide']}")
print(f"Hold: {levels['hold_period']}")
print(f"Buy signals: {signals['buy_signals']}")
print(f"Sell signals: {signals['sell_signals']}")

# Test position diagnosis
diag = engine.diagnose_holding(
    '688017', '半导体龙头', 120.5, 138.5, 25.0, df, fund_flow
)
print(f"Diagnosis: {diag['action']} - {diag['advice'][:80]}...")

# Test backtest
for strat in ['MACD金叉+放量', '多因子综合', '布林带反转', '均线系统']:
    result = run_backtest(df, strat)
    print(f"Backtest [{strat}]: return={result['total_return']:.1f}%, sharpe={result['sharpe_ratio']:.3f}, winrate={result['win_rate']:.1f}%")

# Test all strategies comparison
comp = compare_strategies(df)
print(f"\nStrategy comparison ({len(comp)} strategies):")
for _, row in comp.iterrows():
    print(f"  {row['strategy']:15s} | {row['type']:6s} | return={row['total_return']:6.1f}% | sharpe={row['sharpe_ratio']:5.3f} | WR={row['win_rate']:4.1f}%")

# Test portfolio advice
advice = engine.portfolio_advice([
    {'code': '688017', 'cost': 120, 'position': 150000, 'sector': '半导体设备'},
    {'code': '688981', 'cost': 42, 'position': 80000, 'sector': '存储芯片'},
    {'code': '300750', 'cost': 180, 'position': 60000, 'sector': '新能源'},
], 500000)
print(f"\nPortfolio: cash={advice['cash_pct']:.0f}%, warnings={len(advice['warnings'])}, actions={len(advice['actions'])}")

# Test market sentiment
sent = engine.market_sentiment(df, 87, 12, 1.8)
print(f"Market: {sent['overall']} | style={sent['market_style']} | risk={sent['risk_appetite']}")

print("\n=== ALL TESTS PASSED ===")
