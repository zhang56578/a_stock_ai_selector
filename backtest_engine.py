#!/usr/bin/env python3
"""
A股智选 - 策略回测引擎
支持：自定义策略、多策略对比、绩效指标、可视化
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Callable, Any, Tuple
from datetime import datetime


class BacktestEngine:
    """轻量级回测引擎"""

    def __init__(self, initial_capital: float = 100000.0, commission: float = 0.0003):
        self.initial_capital = initial_capital
        self.capital = initial_capital
        self.commission = commission
        self.position = 0
        self.trades = []
        self.equity_curve = []

    def reset(self):
        self.capital = self.initial_capital
        self.position = 0
        self.trades = []
        self.equity_curve = []

    def buy(self, price: float, date, pct: float = 1.0):
        if self.capital <= 0:
            return
        amount = self.capital * pct * (1 - self.commission)
        shares = int(amount / price / 100) * 100
        if shares > 0:
            cost = shares * price * (1 + self.commission)
            self.capital -= cost
            self.position += shares
            self.trades.append({'date': date, 'action': 'BUY', 'price': price,
                                'shares': shares, 'cost': cost, 'position': self.position})

    def sell(self, price: float, date, pct: float = 1.0):
        if self.position <= 0:
            return
        shares = int(self.position * pct / 100) * 100
        if shares > 0:
            revenue = shares * price * (1 - self.commission)
            self.capital += revenue
            self.position -= shares
            self.trades.append({'date': date, 'action': 'SELL', 'price': price,
                                'shares': shares, 'revenue': revenue, 'position': self.position})

    def total_value(self, current_price: float) -> float:
        return self.capital + self.position * current_price

    def run(self, df: pd.DataFrame, strategy_func: Callable,
            params: Dict = None, verbose: bool = False) -> Dict[str, Any]:
        """执行回测"""
        self.reset()

        if params is None:
            params = {}
        dates = df.index.tolist()

        for i in range(1, len(df)):
            row = df.iloc[i]
            prev_row = df.iloc[i-1]
            date = dates[i]
            price = row['close']

            signal = strategy_func(df.iloc[:i+1], i, params)

            if signal == 'BUY' and self.position == 0:
                self.buy(price, date)
                if verbose:
                    print(f"  [{date}] BUY @ {price:.2f} | 持仓: {self.position}股")

            elif signal == 'SELL' and self.position > 0:
                self.sell(price, date)
                if verbose:
                    print(f"  [{date}] SELL @ {price:.2f} | 现金: {self.capital:.2f}")

            self.equity_curve.append({
                'date': date,
                'equity': self.total_value(price),
                'price': price,
                'position': self.position,
                'capital': self.capital
            })

        # 最后一天强制清仓
        if self.position > 0:
            last_price = df.iloc[-1]['close']
            self.sell(last_price, dates[-1])

        return self._compute_metrics(df)

    def _compute_metrics(self, df: pd.DataFrame) -> Dict[str, Any]:
        curve = pd.DataFrame(self.equity_curve)
        if curve.empty:
            return {'error': '无效回测'}

        final_value = curve['equity'].iloc[-1]
        total_return = (final_value - self.initial_capital) / self.initial_capital

        rets = curve['equity'].pct_change().dropna()
        if len(rets) < 2:
            return {
                'total_return': round(total_return * 100, 2),
                'final_value': round(final_value, 2),
                'trades_count': len(self.trades),
                'error': '回测数据不足'
            }

        # 年化收益
        years = len(curve) / 252
        if years > 0:
            annual_return = (final_value / self.initial_capital) ** (1 / years) - 1
        else:
            annual_return = total_return

        # 最大回撤
        cumulative = curve['equity'] / self.initial_capital
        rolling_max = cumulative.cummax()
        drawdown = (cumulative - rolling_max) / rolling_max
        max_drawdown = drawdown.min()

        # 胜率
        win_trades = 0
        loss_trades = 0
        for i in range(0, len(self.trades)-1, 2):
            if i + 1 < len(self.trades):
                buy_trade = self.trades[i]
                sell_trade = self.trades[i + 1]
                if buy_trade['action'] == 'BUY' and sell_trade['action'] == 'SELL':
                    trade_return = (sell_trade['revenue'] - buy_trade['cost']) / buy_trade['cost']
                    if trade_return > 0:
                        win_trades += 1
                    else:
                        loss_trades += 1

        total_trades = win_trades + loss_trades
        win_rate = win_trades / total_trades * 100 if total_trades > 0 else 0

        # 夏普比率
        excess = rets.mean() - 0.03 / 252
        sharpe = (excess / rets.std() * np.sqrt(252)) if rets.std() > 0 else 0

        # 盈亏比
        if loss_trades > 0 and win_trades > 0:
            avg_win = sum(1 for t in self.trades if t.get('revenue', 0) - t.get('cost', 0) > 0) / max(win_trades, 1)
            avg_loss = sum(1 for t in self.trades if t.get('revenue', 0) - t.get('cost', 0) < 0) / max(loss_trades, 1)
        else:
            avg_win = avg_loss = 0

        return {
            'initial_capital': self.initial_capital,
            'final_value': round(final_value, 2),
            'total_return': round(total_return * 100, 2),
            'annual_return': round(annual_return * 100, 2),
            'max_drawdown': round(max_drawdown * 100, 2),
            'sharpe_ratio': round(sharpe, 3),
            'trades_count': len(self.trades),
            'win_rate': round(win_rate, 1),
            'total_trades': total_trades,
            'equity_curve': curve,
            'trades_list': self.trades
        }


# ==================== 内置经典策略 ====================

def strategy_macd_golden(df: pd.DataFrame, idx: int, params: Dict) -> str:
    """MACD金叉 + 放量策略"""
    row = df.iloc[idx]
    if idx < 2:
        return 'HOLD'
    prev = df.iloc[idx-1]
    if 'MACD_GOLDEN' not in df.columns:
        macd_golden = (row['MACD'] > row['Signal']) and (prev['MACD'] <= prev['Signal'])
    else:
        macd_golden = row['MACD_GOLDEN'] == 1

    volume_surge = row.get('VOLUME_RATIO', 1) > 1.5
    rsi_ok = 30 < row.get('RSI', 50) < 65

    if macd_golden and volume_surge and rsi_ok:
        return 'BUY'

    if row.get('MACD_DEAD', 0) == 1:
        return 'SELL'
    if row.get('RSI', 50) > 75:
        return 'SELL'
    return 'HOLD'


def strategy_ma_cross(df: pd.DataFrame, idx: int, params: Dict) -> str:
    """MA5上穿MA20 + 均线多头策略"""
    row = df.iloc[idx]
    if idx < 2:
        return 'HOLD'
    prev = df.iloc[idx-1]

    ma_bull = row.get('MA_BULL', 0) == 1
    ma_cross = (row.get('MA5', 0) > row.get('MA20', 0)) and (prev.get('MA5', 0) <= prev.get('MA20', 0))

    if ma_cross and ma_bull:
        return 'BUY'

    if row.get('MA_BEAR', 0) == 1 and row.get('MA5', 0) < row.get('MA20', 0):
        return 'SELL'
    if row.get('RSI', 50) > 78:
        return 'SELL'
    return 'HOLD'


def strategy_bollinger_reversal(df: pd.DataFrame, idx: int, params: Dict) -> str:
    """布林带下轨反弹策略"""
    row = df.iloc[idx]
    if idx < 2:
        return 'HOLD'

    at_lower = row.get('BB_BREAK_LOWER', 0) == 1
    rsi_oversold = row.get('RSI', 50) < 35
    kdj_oversold = row.get('J', 50) < 10

    if at_lower and rsi_oversold and kdj_oversold:
        return 'BUY'

    if row.get('BB_BREAK_UPPER', 0) == 1:
        return 'SELL'
    if row.get('RSI', 50) > 72:
        return 'SELL'
    return 'HOLD'


def strategy_multi_factor(df: pd.DataFrame, idx: int, params: Dict) -> str:
    """多因子综合策略"""
    row = df.iloc[idx]
    if idx < 2:
        return 'HOLD'
    prev = df.iloc[idx-1]

    buy_score = 0
    sell_score = 0

    if row.get('MACD_GOLDEN', 0) == 1 or (row['MACD'] > row['Signal'] and prev['MACD'] <= prev['Signal']):
        buy_score += 2
    if row.get('KDJ_GOLDEN', 0) == 1:
        buy_score += 1
    if row.get('MA_BULL', 0) == 1:
        buy_score += 2
    if row.get('RSI', 50) < 35:
        buy_score += 1
    if row.get('VOLUME_RATIO', 1) > 1.5:
        buy_score += 1
    if row.get('WR_OVERSOLD', 0) == 1:
        buy_score += 1

    if row.get('MACD_DEAD', 0) == 1:
        sell_score += 2
    if row.get('MA_BEAR', 0) == 1:
        sell_score += 2
    if row.get('RSI', 50) > 75:
        sell_score += 1
    if row.get('MACD_DIVERGENCE_BEAR', 0) == 1:
        sell_score += 3
    if row.get('SHOOTING_STAR', 0) == 1:
        sell_score += 2

    if buy_score >= 5 and sell_score <= 1:
        return 'BUY'
    if sell_score >= 4:
        return 'SELL'
    return 'HOLD'


def strategy_breakout_volume(df: pd.DataFrame, idx: int, params: Dict) -> str:
    """突破放量策略"""
    row = df.iloc[idx]
    if idx < 20:
        return 'HOLD'
    prev = df.iloc[idx-1]
    prev_20 = df.iloc[idx-20:idx]

    breakout = row['close'] > prev_20['close'].max()
    volume_surge = row.get('VOLUME_RATIO', 1) > 2.0
    trend_up = row.get('TREND_UP', 0) == 1
    rsi_ok = row.get('RSI', 50) < 70

    if breakout and volume_surge and trend_up and rsi_ok:
        return 'BUY'

    if row.get('MACD_DEAD', 0) == 1 and row.get('RSI', 50) > 70:
        return 'SELL'
    if row['close'] < row.get('MA20', 0) and prev['close'] >= prev.get('MA20', 0):
        return 'SELL'
    return 'HOLD'


# 策略注册表
BUILTIN_STRATEGIES = {
    'MACD金叉+放量': {
        'func': strategy_macd_golden,
        'desc': 'MACD金叉且成交量放大1.5倍以上时买入，MACD死叉或RSI>75时卖出',
        'params': {},
        'type': '趋势跟踪'
    },
    '均线系统': {
        'func': strategy_ma_cross,
        'desc': 'MA5上穿MA20且均线多头买入，MA死叉或RSI超买时卖出',
        'params': {},
        'type': '趋势跟踪'
    },
    '布林带反转': {
        'func': strategy_bollinger_reversal,
        'desc': '跌破布林下轨+RIS超卖+KDJ超卖买入，突破上轨卖出',
        'params': {},
        'type': '均值回归'
    },
    '多因子综合': {
        'func': strategy_multi_factor,
        'desc': '综合MACD/KDJ/MA/RSI/WR/成交量等多维度信号做决策',
        'params': {},
        'type': '多因子'
    },
    '突破放量': {
        'func': strategy_breakout_volume,
        'desc': '突破前期高点且放量2倍以上买入，跌破MA20卖出',
        'params': {},
        'type': '突破'
    },
}


def run_backtest(df: pd.DataFrame, strategy_name: str = 'MACD金叉+放量',
                 initial_capital: float = 100000.0) -> Dict[str, Any]:
    """便捷回测函数"""
    engine = BacktestEngine(initial_capital)
    if strategy_name not in BUILTIN_STRATEGIES:
        return {'error': f'未知策略: {strategy_name}'}

    strategy = BUILTIN_STRATEGIES[strategy_name]
    return engine.run(df, strategy['func'], strategy['params'])


def compare_strategies(df: pd.DataFrame) -> pd.DataFrame:
    """对比所有内置策略"""
    results = []
    for name, info in BUILTIN_STRATEGIES.items():
        engine = BacktestEngine()
        result = engine.run(df, info['func'], info['params'])
        result['strategy'] = name
        result['type'] = info['type']
        results.append(result)

    comparison = pd.DataFrame(results)[[
        'strategy', 'type', 'total_return', 'annual_return',
        'max_drawdown', 'sharpe_ratio', 'win_rate', 'trades_count'
    ]]
    return comparison
