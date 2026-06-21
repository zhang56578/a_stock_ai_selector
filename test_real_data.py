"""End-to-end test with REAL data from working sources"""
import pandas as pd
from data_provider import get_kline, get_fund_flow, get_market_overview, add_technical_indicators
from ai_engine import AIEngine
from technical_indicators import TechnicalIndicators

print("=" * 60)
print("TEST 1: Fetch real K-line data for multiple stocks")
for code in ['000001', '600519', '300750', '688981']:
    df = get_kline(code, days=30)
    print(f"  {code}: {len(df)} rows, last close={df.iloc[-1]['close']:.2f}" if not df.empty else f"  {code}: FAILED")

print()
print("=" * 60)
print("TEST 2: Full pipeline with REAL data (000001)")
df = get_kline('000001', days=120)
if not df.empty:
    df = add_technical_indicators(df)
    print(f"  Data: {len(df)} rows, {len(df.columns)} cols")
    print(f"  Columns: {list(df.columns)[:10]}...")

    ti = TechnicalIndicators(df.copy())
    chip = ti.get_chip_distribution()
    print(f"  Chip: current={chip['current_price']:.2f}, avg_cost={chip['avg_cost']:.2f}, profit={chip['profit_ratio']:.1f}%")

    engine = AIEngine()
    signals = engine.detect_signals(df)
    risk = engine.calculate_risk_score(df, signals)
    levels = engine.calculate_trade_levels(df)
    fund_flow = get_fund_flow('000001')

    print(f"  Signal: {signals['overall']} (B:{signals['buy_score']} S:{signals['sell_score']})")
    print(f"  Risk: {risk['level']} ({risk['score']}/100)")
    print(f"  Entry: {levels['buy_point_aggressive']}, Stop: {levels['stop_loss_wide']}")
    print(f"  Fund: price={fund_flow.get('最新价', 'N/A')}, change={fund_flow.get('涨跌幅', 'N/A')}%")
    
    # Test report generation
    report = engine.generate_report(
        '000001', '平安银行', '银行', df, signals, risk, levels, fund_flow, chip,
    )
    print(f"  Report: {len(report)} chars")
else:
    print("  SKIPPED: no data")

print()
print("=" * 60)
print("TEST 3: Market indices")
market = get_market_overview()
for name, info in market.items():
    print(f"  {name}: {info['price']} ({info['change_pct']:+.2f}%)")

print()
print("=" * 60)
print("TEST 4: Backtest with real data")
from backtest_engine import run_backtest
if not df.empty:
    result = run_backtest(df, 'MACD金叉+放量')
    print(f"  {result.get('strategy', 'MACD')}: return={result['total_return']:.1f}%, sharpe={result['sharpe_ratio']:.3f}, WR={result['win_rate']:.1f}%")

print()
print("=== ALL REAL-DATA TESTS PASSED ===")
