#!/usr/bin/env python3
"""
A股智选 - 全维度技术指标计算模块
覆盖：均线、MACD、RSI、KDJ、布林带、ATR、ADX/DMI、OBV、WR、BIAS、CCI、
      筹码分布、量价分析、形态识别
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional


class TechnicalIndicators:
    """一站式技术指标计算器，支持链式调用"""

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._ensure_columns(['open', 'high', 'low', 'close', 'volume'])

    def _ensure_columns(self, cols: list):
        for c in cols:
            if c not in self.df.columns:
                raise ValueError(f"缺少必要列: {c}")

    # ===== 均线系统 =====
    def add_ma(self, periods=(5, 10, 20, 60, 120, 250)):
        for p in periods:
            self.df[f'MA{p}'] = self.df['close'].rolling(p).mean()
        return self

    def add_ema(self, periods=(5, 10, 20, 60)):
        for p in periods:
            self.df[f'EMA{p}'] = self.df['close'].ewm(span=p, adjust=False).mean()
        return self

    def add_ma_system(self):
        """均线多头/空头排列判断"""
        self.add_ma(periods=(5, 10, 20, 60))
        m = self.df
        m['MA_BULL'] = (
            (m['MA5'] > m['MA10']) & (m['MA10'] > m['MA20']) &
            (m['MA20'] > m['MA60'])
        ).astype(int)
        m['MA_BEAR'] = (
            (m['MA5'] < m['MA10']) & (m['MA10'] < m['MA20']) &
            (m['MA20'] < m['MA60'])
        ).astype(int)
        m['MA_CROSS'] = (m['MA5'] > m['MA20']) & (m['MA5'].shift(1) <= m['MA20'].shift(1))
        m['MA_CROSS'] = m['MA_CROSS'].fillna(0).astype(int)
        return self

    # ===== MACD =====
    def add_macd(self, fast=12, slow=26, signal=9):
        m = self.df
        m['EMA_fast'] = m['close'].ewm(span=fast, adjust=False).mean()
        m['EMA_slow'] = m['close'].ewm(span=slow, adjust=False).mean()
        m['MACD'] = m['EMA_fast'] - m['EMA_slow']
        m['Signal'] = m['MACD'].ewm(span=signal, adjust=False).mean()
        m['Histogram'] = m['MACD'] - m['Signal']
        m['MACD_GOLDEN'] = (m['MACD'] > m['Signal']) & (m['MACD'].shift(1) <= m['Signal'].shift(1))
        m['MACD_DEAD'] = (m['MACD'] < m['Signal']) & (m['MACD'].shift(1) >= m['Signal'].shift(1))
        m['MACD_DIVERGENCE_BULL'] = self._macd_bull_divergence(m)
        m['MACD_DIVERGENCE_BEAR'] = self._macd_bear_divergence(m)
        return self

    def _macd_bull_divergence(self, m):
        """底背离：价格新低，MACD未新低"""
        low_10 = m['close'].rolling(10).min()
        macd_low = m['MACD'].rolling(10).min()
        return (m['close'] <= low_10.shift(1)) & (m['MACD'] > macd_low.shift(1))

    def _macd_bear_divergence(self, m):
        """顶背离：价格新高，MACD未新高"""
        high_10 = m['close'].rolling(10).max()
        macd_high = m['MACD'].rolling(10).max()
        return (m['close'] >= high_10.shift(1)) & (m['MACD'] < macd_high.shift(1))

    # ===== RSI =====
    def add_rsi(self, period=14):
        delta = self.df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
        loss = loss.replace(0, np.nan)
        rs = gain / loss
        self.df['RSI'] = 100 - (100 / (1 + rs))
        self.df['RSI_OVERBOUGHT'] = (self.df['RSI'] > 70).astype(int)
        self.df['RSI_OVERSOLD'] = (self.df['RSI'] < 30).astype(int)
        return self

    # ===== KDJ =====
    def add_kdj(self, n=9, m1=3, m2=3):
        low_n = self.df['low'].rolling(n).min()
        high_n = self.df['high'].rolling(n).max()
        rsv = (self.df['close'] - low_n) / (high_n - low_n + 1e-9) * 100
        self.df['K'] = rsv.ewm(com=m1 - 1, adjust=False).mean()
        self.df['D'] = self.df['K'].ewm(com=m2 - 1, adjust=False).mean()
        self.df['J'] = 3 * self.df['K'] - 2 * self.df['D']
        self.df['KDJ_GOLDEN'] = (self.df['K'] > self.df['D']) & (self.df['K'].shift(1) <= self.df['D'].shift(1))
        self.df['KDJ_DEAD'] = (self.df['K'] < self.df['D']) & (self.df['K'].shift(1) >= self.df['D'].shift(1))
        return self

    # ===== 布林带 =====
    def add_bollinger(self, period=20, std=2):
        m = self.df
        m['BB_MIDDLE'] = m['close'].rolling(period).mean()
        bb_std = m['close'].rolling(period).std()
        m['BB_UPPER'] = m['BB_MIDDLE'] + std * bb_std
        m['BB_LOWER'] = m['BB_MIDDLE'] - std * bb_std
        m['BB_WIDTH'] = (m['BB_UPPER'] - m['BB_LOWER']) / m['BB_MIDDLE'] * 100
        m['BB_POSITION'] = (m['close'] - m['BB_LOWER']) / (m['BB_UPPER'] - m['BB_LOWER'] + 1e-9)
        m['BB_SQUEEZE'] = (m['BB_WIDTH'] < m['BB_WIDTH'].rolling(50).mean() * 0.7).astype(int)
        m['BB_BREAK_UPPER'] = (m['close'] > m['BB_UPPER']).astype(int)
        m['BB_BREAK_LOWER'] = (m['close'] < m['BB_LOWER']).astype(int)
        return self

    # ===== ATR 真实波幅 =====
    def add_atr(self, period=14):
        m = self.df
        tr1 = m['high'] - m['low']
        tr2 = (m['high'] - m['close'].shift(1)).abs()
        tr3 = (m['low'] - m['close'].shift(1)).abs()
        m['TR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        m['ATR'] = m['TR'].rolling(period).mean()
        m['ATR_PCT'] = m['ATR'] / m['close'] * 100
        return self

    # ===== ADX / DMI =====
    def add_adx(self, period=14):
        m = self.df
        m['DM_PLUS'] = m['high'].diff().clip(lower=0)
        m['DM_MINUS'] = (-m['low'].diff()).clip(lower=0)
        m['DM_PLUS'] = m['DM_PLUS'].where(
            m['DM_PLUS'] > m['DM_MINUS'], 0)
        m['DM_MINUS'] = m['DM_MINUS'].where(
            m['DM_MINUS'] > m['DM_PLUS'], 0)

        m['TR_SMOOTH'] = m['TR'].rolling(period).sum() if 'TR' in m.columns else \
            m['high'].combine(m['close'].shift(1), max) - \
            m['low'].combine(m['close'].shift(1), min)

        m['DI_PLUS'] = (m['DM_PLUS'].rolling(period).sum() /
                        m['TR'].rolling(period).sum() * 100).fillna(0)
        m['DI_MINUS'] = (m['DM_MINUS'].rolling(period).sum() /
                         m['TR'].rolling(period).sum() * 100).fillna(0)
        m['DX'] = (abs(m['DI_PLUS'] - m['DI_MINUS']) /
                   (m['DI_PLUS'] + m['DI_MINUS'] + 1e-9) * 100)
        m['ADX'] = m['DX'].rolling(period).mean()
        m['ADX_TRENDING'] = (m['ADX'] > 25).astype(int)
        m['ADX_STRONG'] = (m['ADX'] > 40).astype(int)
        return self

    # ===== OBV 能量潮 =====
    def add_obv(self):
        m = self.df
        direction = np.where(m['close'] > m['close'].shift(1), 1,
                             np.where(m['close'] < m['close'].shift(1), -1, 0))
        m['OBV'] = (direction * m['volume']).cumsum()
        m['OBV_MA'] = m['OBV'].rolling(20).mean()
        return self

    # ===== WR 威廉指标 =====
    def add_wr(self, period=14):
        m = self.df
        high_n = m['high'].rolling(period).max()
        low_n = m['low'].rolling(period).min()
        m['WR'] = (high_n - m['close']) / (high_n - low_n + 1e-9) * 100
        m['WR_OVERSOLD'] = (m['WR'] > 80).astype(int)
        m['WR_OVERBOUGHT'] = (m['WR'] < 20).astype(int)
        return self

    # ===== BIAS 乖离率 =====
    def add_bias(self, periods=(6, 12, 24)):
        for p in periods:
            ma = self.df['close'].rolling(p).mean()
            self.df[f'BIAS{p}'] = (self.df['close'] - ma) / ma * 100
        return self

    # ===== CCI 商品通道指数 =====
    def add_cci(self, period=14):
        m = self.df
        tp = (m['high'] + m['low'] + m['close']) / 3
        ma_tp = tp.rolling(period).mean()
        md = tp.rolling(period).apply(lambda x: np.abs(x - x.mean()).mean())
        m['CCI'] = (tp - ma_tp) / (0.015 * md + 1e-9)
        return self

    # ===== 成交量分析 =====
    def add_volume_analysis(self, period=20):
        m = self.df
        m['VOLUME_MA5'] = m['volume'].rolling(5).mean()
        m['VOLUME_MA20'] = m['volume'].rolling(period).mean()
        m['VOLUME_RATIO'] = m['volume'] / m['VOLUME_MA20'].replace(0, np.nan)
        m['VOLUME_SURGE'] = (m['VOLUME_RATIO'] > 2).astype(int)
        m['VOLUME_SHRINK'] = (m['VOLUME_RATIO'] < 0.5).astype(int)
        m['VOLUME_TREND'] = m['VOLUME_MA5'] / m['VOLUME_MA20'].shift(1)
        return self

    # ===== 筹码分布估算 =====
    def add_chip_analysis(self, window=60):
        """简化筹码分布：成本集中度、获利比例、平均成本"""
        m = self.df
        m['CHIP_AVG_COST'] = m['close'].rolling(window).mean()
        m['CHIP_PEAK'] = m['close'].rolling(window).apply(
            lambda x: np.histogram(x, bins=10)[1][np.argmax(np.histogram(x, bins=10)[0])]
        )
        m['CHIP_HIGH_CONC'] = (m['close'].rolling(window).std() /
                               m['CHIP_AVG_COST'] * 100 < 15).astype(int)
        m['PROFIT_RATIO'] = (m['close'].rolling(window).apply(
            lambda x: (x > x.iloc[-1]).sum() / len(x) * 100
        ))
        return self

    def get_chip_distribution(self, bins=50):
        """获取详细筹码分布数据（用于绘图）"""
        prices = self.df['close'].dropna().values[-120:]
        if len(prices) < 20:
            return None

        hist, edges = np.histogram(prices, bins=bins, weights=np.ones_like(prices))
        centers = (edges[:-1] + edges[1:]) / 2

        return {
            'prices': centers.tolist(),
            'chips': hist.tolist(),
            'current_price': float(prices[-1]),
            'max_chip_price': float(centers[np.argmax(hist)]),
            'avg_cost': float(prices.mean()),
            'cost_std': float(prices.std()),
            'profit_ratio': float((prices > prices[-1]).sum() / len(prices) * 100)
        }

    # ===== 价格形态 =====
    def add_patterns(self):
        m = self.df
        o, h, l, c = m['open'], m['high'], m['low'], m['close']

        body = abs(c - o)
        upper_shadow = h - np.maximum(o, c)
        lower_shadow = np.minimum(o, c) - l

        m['HAMMER'] = (
            (lower_shadow > body * 2) & (upper_shadow < body * 0.3) &
            (body > 0)
        ).astype(int)

        m['SHOOTING_STAR'] = (
            (upper_shadow > body * 2) & (lower_shadow < body * 0.3) &
            (body > 0)
        ).astype(int)

        m['DOJI'] = (body / (h - l + 1e-9) < 0.1).astype(int)

        engulf_bull = (c > o) & (c.shift(1) < o.shift(1)) & (c > o.shift(1)) & (o < c.shift(1))
        m['BULLISH_ENGULF'] = engulf_bull.astype(int)

        engulf_bear = (c < o) & (c.shift(1) > o.shift(1)) & (c < o.shift(1)) & (o > c.shift(1))
        m['BEARISH_ENGULF'] = engulf_bear.astype(int)

        return self

    # ===== 趋势强度 =====
    def add_trend_strength(self, period=20):
        m = self.df
        m['RET'] = m['close'].pct_change()
        m['TREND'] = m['close'].rolling(period).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0]
        )
        m['TREND_STRENGTH'] = m['TREND'] / m['close'] * 100 * period
        m['TREND_UP'] = (m['TREND'] > 0).astype(int)
        m['VOLATILITY'] = m['RET'].rolling(period).std() * np.sqrt(252)
        m['SHARPE'] = m['RET'].rolling(period).mean() / \
            m['RET'].rolling(period).std().replace(0, np.nan) * np.sqrt(252)
        return self

    # ===== 综合主力资金足迹（基于量价推算） =====
    def add_money_flow(self, period=14):
        m = self.df
        tp = (m['high'] + m['low'] + m['close']) / 3
        raw_mf = tp * m['volume']
        pos_mf = np.where(tp > tp.shift(1), raw_mf, 0)
        neg_mf = np.where(tp < tp.shift(1), raw_mf, 0)
        pos_sum = pd.Series(pos_mf).rolling(period).sum()
        neg_sum = pd.Series(neg_mf).rolling(period).sum()
        m['MFI'] = 100 - 100 / (1 + pos_sum / (neg_sum + 1e-9))
        return self

    # ===== 涨跌停信号 =====
    def add_limit_signals(self, limit_pct=0.098):
        m = self.df
        m['Limit_Up'] = (m['close'].pct_change() > limit_pct).astype(int)
        m['Limit_Down'] = (m['close'].pct_change() < -limit_pct).astype(int)
        return self

    # ===== 一键全指标 =====
    def add_all(self):
        return (self
                .add_ma_system()
                .add_ema()
                .add_macd()
                .add_rsi()
                .add_kdj()
                .add_bollinger()
                .add_atr()
                .add_adx()
                .add_obv()
                .add_wr()
                .add_bias()
                .add_cci()
                .add_volume_analysis()
                .add_chip_analysis()
                .add_patterns()
                .add_trend_strength()
                .add_money_flow()
                .add_limit_signals())

    def result(self) -> pd.DataFrame:
        core_cols = ['open', 'high', 'low', 'close', 'volume']
        return self.df.dropna(subset=core_cols)


def compute_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """快捷函数：对DataFrame一键添加所有技术指标"""
    ti = TechnicalIndicators(df)
    return ti.add_all().result()


def add_indicators_streamlined(df: pd.DataFrame) -> pd.DataFrame:
    """精简版：仅核心指标（用于实时快速计算）"""
    ti = TechnicalIndicators(df)
    return (ti
            .add_ma_system()
            .add_macd()
            .add_rsi()
            .add_kdj()
            .add_bollinger()
            .add_atr()
            .add_volume_analysis()
            .result())
