#!/usr/bin/env python3
"""
A股智选 - AI智能分析引擎
基于多维度规则 + 可扩展LLM的自然语言分析引擎
负责：买卖点识别、风险评分、持仓诊断、策略建议、研报生成
"""

import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

# 尝试导入LLM库（可选）
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import google.generativeai as genai
    HAS_GEMINI = True
except ImportError:
    HAS_GEMINI = False


class AIEngine:
    """
    AI分析引擎
    规则引擎 + 可扩展LLM = 智能分析结果
    """

    def __init__(self, use_llm: bool = False, llm_provider: str = "openai", api_key: str = None):
        self.use_llm = use_llm
        self.llm_provider = llm_provider
        self.api_key = api_key
        if use_llm and api_key:
            self._init_llm()

    def _init_llm(self):
        if self.llm_provider == "openai" and HAS_OPENAI:
            openai.api_key = self.api_key
        elif self.llm_provider == "gemini" and HAS_GEMINI:
            genai.configure(api_key=self.api_key)

    # ============================================================
    #  核心一：买卖点识别
    # ============================================================
    def detect_signals(self, df: pd.DataFrame) -> Dict[str, Any]:
        """多维度综合买卖信号识别（增强梯度版）"""
        latest = df.iloc[-1]
        signals = {
            'buy_signals': [],
            'sell_signals': [],
            'neutral_signals': [],
            'buy_score': 0,
            'sell_score': 0,
            'overall': 'NEUTRAL',
            'confidence': 0.0,
            'details': {}
        }

        # ── MACD ──
        if 'MACD' in df.columns and 'Signal' in df.columns:
            macd_val = latest['MACD'] - latest['Signal']
            if latest.get('MACD_GOLDEN', 0) == 1:
                signals['buy_signals'].append('MACD金叉')
                signals['buy_score'] += 20
            if latest.get('MACD_DEAD', 0) == 1:
                signals['sell_signals'].append('MACD死叉')
                signals['sell_score'] += 20
            if latest.get('MACD_DIVERGENCE_BULL', 0) == 1:
                signals['buy_signals'].append('MACD底背离（强烈看涨）')
                signals['buy_score'] += 35
            if latest.get('MACD_DIVERGENCE_BEAR', 0) == 1:
                signals['sell_signals'].append('MACD顶背离（强烈看跌）')
                signals['sell_score'] += 35
            if latest['MACD'] > latest['Signal']:
                signals['details']['macd_status'] = '多头'
                bonus = min(12, max(1, int((latest['MACD'] - latest['Signal']) / latest['close'] * 10000)))
                signals['buy_score'] += bonus
            else:
                signals['details']['macd_status'] = '空头'
                bonus = min(12, max(1, int((latest['Signal'] - latest['MACD']) / latest['close'] * 10000)))
                signals['sell_score'] += bonus
            # MACD柱子加速
            if len(df) >= 3:
                hist_cur = df['Histogram'].iloc[-1]
                hist_prev = df['Histogram'].iloc[-2]
                hist_pp = df['Histogram'].iloc[-3]
                if hist_cur > hist_prev > hist_pp and hist_cur > 0:
                    signals['buy_signals'].append('MACD红柱放大（动能增强）')
                    signals['buy_score'] += 10
                if hist_cur < hist_prev < hist_pp and hist_cur < 0:
                    signals['sell_signals'].append('MACD绿柱放大（动能衰减）')
                    signals['sell_score'] += 10

        # ── RSI（连续梯度）──
        if 'RSI' in df.columns:
            rsi = latest['RSI']
            signals['details']['rsi'] = round(rsi, 1)
            if rsi <= 20:
                signals['buy_signals'].append('RSI极度超卖(≤20)')
                signals['buy_score'] += 25
            elif rsi <= 28:
                signals['buy_signals'].append('RSI严重超卖(≤28)')
                signals['buy_score'] += 18
            elif rsi <= 35:
                signals['buy_signals'].append('RSI超卖(≤35)')
                signals['buy_score'] += 10
            elif rsi >= 85:
                signals['sell_signals'].append('RSI极度超买(≥85)')
                signals['sell_score'] += 25
            elif rsi >= 75:
                signals['sell_signals'].append('RSI严重超买(≥75)')
                signals['sell_score'] += 15
            elif rsi >= 68:
                signals['sell_signals'].append('RSI偏高(≥68)')
                signals['sell_score'] += 8
            # RSI趋势
            if len(df) >= 5:
                rsi_5d_ago = df['RSI'].iloc[-5]
                if rsi > rsi_5d_ago + 8:
                    signals['buy_signals'].append('RSI持续回升')
                    signals['buy_score'] += 5
                elif rsi < rsi_5d_ago - 8:
                    signals['sell_signals'].append('RSI持续回落')
                    signals['sell_score'] += 5

        # ── KDJ ──
        if all(k in df.columns for k in ['K', 'D', 'J']):
            if latest.get('KDJ_GOLDEN', 0) == 1:
                signals['buy_signals'].append('KDJ金叉')
                signals['buy_score'] += 12
            if latest.get('KDJ_DEAD', 0) == 1:
                signals['sell_signals'].append('KDJ死叉')
                signals['sell_score'] += 12
            j = latest['J']
            signals['details']['kdj_j'] = round(j, 1)
            if j < -5:
                signals['buy_signals'].append('KDJ-J值<0（极度超卖）')
                signals['buy_score'] += 18
            elif j < 5:
                signals['buy_signals'].append('KDJ-J值低（超卖区）')
                signals['buy_score'] += 10
            if j > 105:
                signals['sell_signals'].append('KDJ-J值>105（极度超买）')
                signals['sell_score'] += 15
            elif j > 95:
                signals['sell_signals'].append('KDJ-J值高（超买区）')
                signals['sell_score'] += 8

        # ── 布林带 ──
        if 'BB_POSITION' in df.columns:
            if latest.get('BB_SQUEEZE', 0) == 1:
                signals['neutral_signals'].append('布林带收口（变盘前兆）')
                signals['buy_score'] += 3
                signals['sell_score'] += 3
            if latest.get('BB_BREAK_LOWER', 0) == 1:
                signals['buy_signals'].append('跌破布林下轨（超跌反弹）')
                signals['buy_score'] += 12
            if latest.get('BB_BREAK_UPPER', 0) == 1:
                signals['sell_signals'].append('突破布林上轨（减仓信号）')
                signals['sell_score'] += 10
            # BB位置梯度
            bb_pos = latest['BB_POSITION']
            if 0 <= bb_pos < 0.1:
                signals['buy_signals'].append('价格位于布林底部')
                signals['buy_score'] += 8

        # ── 均线系统 ──
        if 'MA_BULL' in df.columns and latest.get('MA_BULL', 0) == 1:
            signals['buy_signals'].append('均线多头排列(MA5>10>20>60)')
            signals['buy_score'] += 20
        if 'MA_BEAR' in df.columns and latest.get('MA_BEAR', 0) == 1:
            signals['sell_signals'].append('均线空头排列')
            signals['sell_score'] += 20
        if latest.get('MA_CROSS', 0) == 1:
            signals['buy_signals'].append('MA5上穿MA20')
            signals['buy_score'] += 12
        # 价格相对MA20/MA60位置
        if 'MA20' in df.columns and 'MA60' in df.columns:
            pct_ma20 = (latest['close'] - latest['MA20']) / latest['MA20'] * 100
            pct_ma60 = (latest['close'] - latest['MA60']) / latest['MA60'] * 100
            if pct_ma20 > 15:
                signals['sell_signals'].append(f'偏离MA20达{pct_ma20:.0f}%（高位）')
                signals['sell_score'] += 8
            elif pct_ma20 < -10:
                signals['buy_signals'].append(f'偏离MA20达{pct_ma20:.0f}%（低位）')
                signals['buy_score'] += 8
            if pct_ma60 > 20:
                signals['sell_score'] += 5
            elif pct_ma60 < -15:
                signals['buy_score'] += 5

        # ── 量价关系 ──
        if 'volume' in df.columns and len(df) >= 5:
            vol_ma5 = df['volume'].tail(5).mean()
            vol_latest = latest['volume']
            vol_ratio = vol_latest / vol_ma5 if vol_ma5 > 0 else 1
            price_chg = (latest['close'] - df['close'].iloc[-2]) / df['close'].iloc[-2] * 100
            if vol_ratio > 2.0 and price_chg > 2:
                signals['buy_signals'].append('放量上涨（量价配合）')
                signals['buy_score'] += 12
            elif vol_ratio > 1.5 and price_chg > 0:
                signals['buy_signals'].append('温和放量上涨')
                signals['buy_score'] += 6
            elif vol_ratio > 1.5 and price_chg < -2:
                signals['sell_signals'].append('放量下跌（量价背离）')
                signals['sell_score'] += 12
            elif vol_ratio < 0.5 and price_chg > 0:
                signals['sell_signals'].append('缩量上涨（上攻乏力）')
                signals['sell_score'] += 5
            elif vol_ratio < 0.5 and price_chg < 0:
                signals['buy_signals'].append('缩量下跌（抛压减轻）')
                signals['buy_score'] += 5

        # ── ADX趋势强度 ──
        if 'ADX' in df.columns:
            adx = latest['ADX']
            if adx > 40:
                signals['details']['adx_trend'] = '强趋势'
                if 'DI_PLUS' in df.columns and latest['DI_PLUS'] > latest['DI_MINUS']:
                    signals['buy_signals'].append(f'ADX强趋势+DI+向上({adx:.0f})')
                    signals['buy_score'] += 10
                else:
                    signals['sell_signals'].append(f'ADX强趋势+DI-向下({adx:.0f})')
                    signals['sell_score'] += 10
            elif adx > 25:
                signals['details']['adx_trend'] = '趋势明确'
                if 'DI_PLUS' in df.columns and latest['DI_PLUS'] > latest['DI_MINUS']:
                    signals['buy_score'] += 5
                else:
                    signals['sell_score'] += 5
            else:
                signals['details']['adx_trend'] = '震荡'

        # ── WR ──
        if 'WR' in df.columns:
            wr = latest['WR']
            if wr > 90:
                signals['buy_signals'].append(f'WR极度超卖({wr:.0f})')
                signals['buy_score'] += 10
            elif wr > 75:
                signals['buy_signals'].append(f'WR超卖({wr:.0f})')
                signals['buy_score'] += 6
            elif wr < 10:
                signals['sell_signals'].append(f'WR极度超买({wr:.0f})')
                signals['sell_score'] += 10
            elif wr < 25:
                signals['sell_signals'].append(f'WR超买({wr:.0f})')
                signals['sell_score'] += 6

        # ── CCI ──
        if 'CCI' in df.columns:
            cci = latest['CCI']
            if cci < -200:
                signals['buy_signals'].append(f'CCI<-200（极度超卖）')
                signals['buy_score'] += 8
            elif cci > 200:
                signals['sell_signals'].append(f'CCI>200（极度超买）')
                signals['sell_score'] += 8

        # ── 动量/涨跌幅 ──
        if len(df) >= 5:
            ret_5d = (latest['close'] / df['close'].iloc[-5] - 1) * 100
            if ret_5d > 15:
                signals['sell_signals'].append(f'5日涨幅>15%（短期过热）')
                signals['sell_score'] += 10
            elif ret_5d < -10:
                signals['buy_signals'].append(f'5日跌幅>10%（超跌机会）')
                signals['buy_score'] += 10
        if len(df) >= 20:
            ret_20d = (latest['close'] / df['close'].iloc[-20] - 1) * 100
            signals['details']['ret_20d'] = round(ret_20d, 1)
            if ret_20d > 30:
                signals['sell_signals'].append(f'20日涨幅>30%（高位风险）')
                signals['sell_score'] += 15
            elif ret_20d < -20:
                signals['buy_signals'].append(f'20日跌幅>20%（超跌区域）')
                signals['buy_score'] += 15

        # ── K线形态 ──
        if len(df) >= 2:
            body = abs(latest['close'] - latest['open'])
            shadow_bottom = min(latest['open'], latest['close']) - latest['low']
            shadow_top = latest['high'] - max(latest['open'], latest['close'])
            total_range = latest['high'] - latest['low']
            if total_range > 0:
                if body < total_range * 0.2 and shadow_bottom > total_range * 0.5:
                    signals['buy_signals'].append('长下影线（锤子/探底）')
                    signals['buy_score'] += 8
                if body < total_range * 0.2 and shadow_top > total_range * 0.5:
                    signals['sell_signals'].append('长上影线（射击之星）')
                    signals['sell_score'] += 8

        # ── 计算综合评分 ──
        total = signals['buy_score'] + signals['sell_score']
        if total > 0:
            net = (signals['buy_score'] - signals['sell_score']) / total
            signals['confidence'] = abs(net)
            if signals['buy_score'] >= 50:
                signals['overall'] = 'STRONG_BUY'
            elif signals['buy_score'] >= 30:
                signals['overall'] = 'BUY'
            elif signals['sell_score'] >= 50:
                signals['overall'] = 'STRONG_SELL'
            elif signals['sell_score'] >= 30:
                signals['overall'] = 'SELL'
            elif signals['buy_score'] > signals['sell_score']:
                signals['overall'] = 'WEAK_BUY'
            elif signals['sell_score'] > signals['buy_score']:
                signals['overall'] = 'WEAK_SELL'
            else:
                signals['overall'] = 'NEUTRAL'

        return signals

    # ============================================================
    #  核心二：买卖价位计算
    # ============================================================
    def calculate_trade_levels(self, df: pd.DataFrame) -> Dict[str, Any]:
        """基于技术分析计算买卖价位、止损止盈"""
        latest = df.iloc[-1]
        price = latest['close']

        atr = latest.get('ATR', price * 0.03)
        support = latest.get('MA20', latest.get('BB_LOWER', price * 0.95))
        resistance = latest.get('BB_UPPER', price * 1.1)

        # 寻找近期支撑/阻力
        recent = df.tail(30)
        levels = {
            'current_price': round(price, 2),
            'atr': round(atr, 2),
            'buy_point_aggressive': round(support, 2),
            'buy_point_conservative': round(support * 0.97, 2),
            'buy_point_bottom_fishing': round(min(recent['low'].min(), price * 0.85), 2),
            'take_profit_1': round(price + atr * 2, 2),
            'take_profit_2': round(price + atr * 3, 2),
            'take_profit_3': round(price * 1.20, 2),
            'stop_loss_tight': round(price - atr * 2, 2),
            'stop_loss_normal': round(price - atr * 2.5, 2),
            'stop_loss_wide': round(min(price * 0.92, support * 0.95), 2),
            'ma5': round(latest.get('MA5', 0), 2),
            'ma20': round(latest.get('MA20', 0), 2),
            'ma60': round(latest.get('MA60', 0), 2),
        }
        levels['stop_loss_max'] = round(min(levels['stop_loss_wide'], price * 0.92), 2)

        # 持仓周期建议
        if 'MA_BULL' in df.columns and latest.get('MA_BULL', 0) == 1:
            levels['hold_period'] = '波段1-3个月 / 中线3-6个月'
        elif 'ADX' in df.columns and latest['ADX'] > 40:
            levels['hold_period'] = '趋势强劲，顺势持有1-3个月'
        else:
            levels['hold_period'] = '短线1-4周，快进快出'

        return levels

    # ============================================================
    #  核心三：风险评分
    # ============================================================
    def calculate_risk_score(self, df: pd.DataFrame, signals: Dict) -> Dict[str, Any]:
        """多维度风险综合评分（0-100，越高越危险）"""
        risk_score = 0
        risk_factors = []
        latest = df.iloc[-1]

        # 波动率风险
        if 'VOLATILITY' in df.columns:
            vol = latest['VOLATILITY']
            if vol > 0.6:
                risk_score += 20
                risk_factors.append(f'高波动率({vol:.1%})')
            elif vol > 0.4:
                risk_score += 10
                risk_factors.append(f'中高波动率({vol:.1%})')

        # 趋势风险
        if 'TREND_DOWN' in df.columns and latest.get('TREND_DOWN', 0):
            risk_score += 15
            risk_factors.append('下降趋势')

        # 超买风险
        if 'RSI' in df.columns and latest['RSI'] > 80:
            risk_score += 15
            risk_factors.append('严重超买(RSI>80)')

        # 成交量异常
        if 'VOLUME_SURGE' in df.columns and latest.get('VOLUME_SURGE', 0):
            risk_score += 8
            risk_factors.append('异常放量')

        # 偏离均线
        if 'BIAS24' in df.columns:
            bias = abs(latest['BIAS24'])
            if bias > 20:
                risk_score += 12
                risk_factors.append(f'严重偏离均线(BIAS={bias:.1f}%)')

        # 筹码分布
        if 'PROFIT_RATIO' in df.columns:
            profit = latest['PROFIT_RATIO']
            if profit > 90:
                risk_score += 10
                risk_factors.append(f'90%+获利盘(获利止盈压力)')
            elif profit < 10:
                risk_score += 5
                risk_factors.append(f'<10%获利盘(套牢盘重)')

        # 价格形态风险
        for pattern, risk_val, desc in [
            ('SHOOTING_STAR', 10, '射击之星（看跌）'),
            ('BEARISH_ENGULF', 12, '看跌吞没'),
        ]:
            if pattern in df.columns and latest.get(pattern, 0) == 1:
                risk_score += risk_val
                risk_factors.append(desc)

        # 归一化
        risk_score = min(risk_score, 100)
        if risk_score >= 70:
            level = '高风险'
            color = '#ff4444'
        elif risk_score >= 40:
            level = '中等风险'
            color = '#ffaa00'
        elif risk_score >= 20:
            level = '中低风险'
            color = '#44aa44'
        else:
            level = '低风险'
            color = '#00cc00'

        return {
            'score': risk_score,
            'level': level,
            'color': color,
            'factors': risk_factors,
        }

    # ============================================================
    #  核心四：持仓诊断
    # ============================================================
    def diagnose_holding(self, code: str, name: str, cost: float,
                         current: float, position_pct: float,
                         df: pd.DataFrame, fund_flow: Dict) -> Dict[str, Any]:
        """持仓综合诊断"""
        signals = self.detect_signals(df)
        risk = self.calculate_risk_score(df, signals)
        levels = self.calculate_trade_levels(df)
        pnl_pct = (current - cost) / cost * 100

        # 诊断逻辑
        diagnosis = {
            'code': code,
            'name': name,
            'cost': round(cost, 2),
            'current': round(current, 2),
            'pnl_pct': round(pnl_pct, 2),
            'position_pct': round(position_pct, 2),
            'signal': signals['overall'],
            'risk': risk,
            'levels': levels,
            'action': '',
            'advice': '',
            'details': []
        }

        # 综合决策
        if pnl_pct > 30:
            diagnosis['action'] = '减仓止盈'
            diagnosis['advice'] = f'浮盈{pnl_pct:.1f}%，建议减仓30-50%锁定利润，剩余仓位观察'
            diagnosis['details'].append('浮盈超过30%，分批止盈策略')
        elif pnl_pct > 15:
            diagnosis['action'] = '部分止盈'
            diagnosis['advice'] = f'浮盈{pnl_pct:.1f}%，建议减仓20-30%，上移止损位至成本价+5%'
            diagnosis['details'].append('浮盈15-30%，适度止盈保护利润')
        elif pnl_pct > 5:
            if signals['overall'] in ['STRONG_BUY', 'BUY']:
                diagnosis['action'] = '继续持有'
                diagnosis['advice'] = '技术面强势，浮盈中，继续持有。回调MA20可加仓'
                diagnosis['details'].append('趋势良好，持股不动')
            elif signals['overall'] in ['STRONG_SELL', 'SELL', 'WEAK_SELL']:
                diagnosis['action'] = '止盈离场'
                diagnosis['advice'] = '技术面转弱建议止盈，锁定利润'
                diagnosis['details'].append('卖出信号出现，见好就收')
            else:
                diagnosis['action'] = '持有观望'
                diagnosis['advice'] = '震荡区间，持有为主，关注突破方向'
                diagnosis['details'].append('方向不明，等待信号')
        elif pnl_pct > -5:
            if signals['overall'] in ['STRONG_BUY', 'BUY']:
                diagnosis['action'] = '持有或小补'
                diagnosis['advice'] = '技术面转好，可小仓位补仓（10%）或继续持有'
                diagnosis['details'].append('买入信号出现，可考虑加仓')
            elif signals['overall'] in ['STRONG_SELL', 'SELL']:
                diagnosis['action'] = '止损/减仓'
                diagnosis['advice'] = '技术面恶化，建议减仓或止损'
                diagnosis['details'].append('卖出信号，控制风险')
            else:
                diagnosis['action'] = '观望'
                diagnosis['advice'] = '窄幅震荡，继续观察，不宜操作'
                diagnosis['details'].append('震荡观望')
        elif pnl_pct > -15:
            if signals['overall'] in ['STRONG_BUY', 'BUY']:
                diagnosis['action'] = '补仓摊薄'
                diagnosis['advice'] = f'浮亏{abs(pnl_pct):.1f}%，技术面转好，可分批补仓'
                diagnosis['details'].append('买入信号+浮亏适中，补仓机会')
            elif signals['overall'] in ['STRONG_SELL']:
                diagnosis['action'] = '止损换股'
                diagnosis['advice'] = '强烈卖出信号，建议止损，换入强势板块个股'
                diagnosis['details'].append('卖出信号强烈，果断止损')
            else:
                diagnosis['action'] = '减仓观望'
                diagnosis['advice'] = '信号不明确，建议减仓等方向明朗'
                diagnosis['details'].append('方向不明，减仓降低风险')
        else:
            diagnosis['action'] = '严格止损或换股'
            diagnosis['advice'] = f'浮亏{abs(pnl_pct):.1f}%严重，严格止损或换强势板块补亏'
            diagnosis['details'].append('深度套牢，考虑换股解套')

        # 加入资金流因素
        if fund_flow.get('主力净流入(亿)', 0) > 5:
            diagnosis['details'].append('主力大幅净流入，加分')
        elif fund_flow.get('主力净流入(亿)', 0) < -3:
            diagnosis['details'].append('主力净流出，减分')

        return diagnosis

    # ============================================================
    #  核心五：综合研报生成
    # ============================================================
    def generate_report(self, code: str, name: str, sector: str,
                        df: pd.DataFrame, signals: Dict, risk: Dict,
                        levels: Dict, fund_flow: Dict, chip: Dict = None,
                        us_link: Dict = None, macro: Dict = None) -> str:
        """生成结构化深度研报（Markdown）"""
        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        change = (latest['close'] - prev['close']) / prev['close'] * 100

        report = f"""
# {code} {name} 深度研究报告
> 生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
> 所属板块: {sector}

---

## 一、综合评级

| 维度 | 评级 | 说明 |
|------|------|------|
| 技术信号 | **{signals['overall']}** | 综合评分: B={signals['buy_score']} S={signals['sell_score']} |
| 风险等级 | **{risk['level']}** | 风险评分: {risk['score']}/100 |
| 资金流向 | {'**流入**' if fund_flow.get('主力净流入(亿)', 0) > 0 else '**流出**'} | 主力净额: {fund_flow.get('主力净流入(亿)', 0)}亿 |

---

## 二、技术面深度分析

### 2.1 关键价位
| 指标 | 数值 | 指标 | 数值 |
|------|------|------|------|
| 最新价 | **{levels['current_price']}** | 今日涨跌 | {change:+.2f}% |
| MA5 | {levels['ma5']} | MA20 | {levels['ma20']} |
| MA60 | {levels['ma60']} | ATR | {levels['atr']} |
| 激进买入 | {levels['buy_point_aggressive']} | 保守买入 | {levels['buy_point_conservative']} |
| 止盈1 | {levels['take_profit_1']} | 止盈2 | {levels['take_profit_2']} |
| 止损紧 | {levels['stop_loss_tight']} | 止损宽 | {levels['stop_loss_wide']} |

### 2.2 技术指标全景
| 指标 | 数值 | 信号 |
|------|------|------|
| MACD | {latest.get('MACD', 0):.3f} | {'金叉' if signals['buy_score']>signals['sell_score'] else '死叉'} |
| RSI(14) | {latest.get('RSI', 0):.1f} | {'超买' if latest.get('RSI', 0)>70 else '超卖' if latest.get('RSI', 0)<30 else '中性'} |
| K | {latest.get('K', 0):.1f} / D | {latest.get('D', 0):.1f} / J | {latest.get('J', 0):.1f} |
| ADX | {latest.get('ADX', 0):.1f} | {'强趋势' if latest.get('ADX', 0)>40 else '趋势' if latest.get('ADX', 0)>25 else '震荡'} |
| WR | {latest.get('WR', 0):.1f} | - |
| CCI | {latest.get('CCI', 0):.1f} | - |
| OBV | {latest.get('OBV', 0):.0f} | OBV_MA: {latest.get('OBV_MA', 0):.0f} |
| 成交量比 | {latest.get('VOLUME_RATIO', 0):.1f}x | {'放量' if latest.get('VOLUME_RATIO', 0)>1.5 else '正常' if latest.get('VOLUME_RATIO', 0)>0.5 else '缩量'} |
| 波动率 | {latest.get('VOLATILITY', 0):.1%} | - |
"""

        # 买卖信号
        if signals['buy_signals']:
            report += "\n### 2.3 买入信号\n"
            for s in signals['buy_signals']:
                report += f"- ✅ {s}\n"

        if signals['sell_signals']:
            report += "\n### 2.3 卖出信号\n"
            for s in signals['sell_signals']:
                report += f"- ❌ {s}\n"

        if signals['neutral_signals']:
            report += "\n### 2.4 中性/关注信号\n"
            for s in signals['neutral_signals']:
                report += f"- ⚠️ {s}\n"

        # 布林带
        report += f"""
### 2.5 布林带
- 上轨: {latest.get('BB_UPPER', 0):.2f}
- 中轨: {latest.get('BB_MIDDLE', 0):.2f}
- 下轨: {latest.get('BB_LOWER', 0):.2f}
- 带宽: {latest.get('BB_WIDTH', 0):.1f}%
- 位置: {latest.get('BB_POSITION', 0):.1%}
"""

        # 筹码分布
        if chip:
            report += f"""
---

## 三、筹码分布
- 当前价格: **{chip.get('current_price', 0):.2f}**
- 筹码密集区: **{chip.get('max_chip_price', 0):.2f}**
- 平均成本: {chip.get('avg_cost', 0):.2f}
- 获利比例: {chip.get('profit_ratio', 0):.1f}%
- 筹码集中度: {'高(成本集中)' if chip.get('cost_std', 0)/chip.get('avg_cost', 1)*100 < 10 else '中等' if chip.get('cost_std', 0)/chip.get('avg_cost', 1)*100 < 20 else '分散'}
"""

        # 资金面
        report += f"""
---

## 四、资金面
- 主力净流入: {fund_flow.get('主力净流入(亿)', 'N/A')} 亿
- 大单净流入: {fund_flow.get('大单净流入(亿)', 'N/A')} 亿
- 北向资金: {fund_flow.get('北向资金(亿)', 'N/A')} 亿
- 趋势: {fund_flow.get('趋势', 'N/A')}
"""

        # 美股联动
        if us_link:
            report += f"""
---

## 五、美股联动参考
| 美股代码 | 昨日涨跌 | 关联A股 |
|----------|----------|----------|
"""
            for ticker, info in us_link.items():
                if isinstance(info, dict):
                    chg = info.get('yesterday_change', info.get('change', 0))
                    related = info.get('a_stock_related', '')
                    report += f"| {ticker} | {chg:+.2f}% | {related} |\n"
                elif isinstance(info, (int, float)):
                    report += f"| {ticker} | {info:+.2f}% | - |\n"
                else:
                    report += f"| {ticker} | {info} | - |\n"

        # 宏观
        if macro:
            report += f"""
---

## 六、宏观参考
"""
            for k, v in macro.items():
                report += f"- {k}: {v}\n"

        # 综合建议
        report += f"""
---

## 七、操作建议

### 买入策略
- 激进: 当前价位附近（{levels['buy_point_aggressive']}）分批建仓
- 保守: 回调至 MA20（{levels['ma20']}）附近建仓
- 抄底: 若回调至 {levels['buy_point_bottom_fishing']}，可重仓

### 卖出/止盈
- 目标1: {levels['take_profit_1']} (+{((levels['take_profit_1']-levels['current_price'])/levels['current_price']*100):.1f}%)
- 目标2: {levels['take_profit_2']} (+{((levels['take_profit_2']-levels['current_price'])/levels['current_price']*100):.1f}%)
- 长线目标: {levels['take_profit_3']}

### 止损
- 严格止损: {levels['stop_loss_tight']} (-{((levels['current_price']-levels['stop_loss_tight'])/levels['current_price']*100):.1f}%)
- 宽止损: {levels['stop_loss_wide']} (-{((levels['current_price']-levels['stop_loss_wide'])/levels['current_price']*100):.1f}%)

### 持仓周期
- **{levels['hold_period']}**

---

## ⚠️ 风险提示
"""
        for f in risk['factors']:
            report += f"- {f}\n"

        report += f"""
- 本报告仅供参考，不构成投资建议，股市有风险，投资需谨慎。
- 建议结合基本面（PE/PB/ROE/成长性）做综合判断。
- 建议严格控制单票仓位不超过总仓位30%。

---

*报告由 A股智选AI引擎 自动生成 | {datetime.now().strftime('%Y-%m-%d')}*
"""
        return report

    # ============================================================
    #  核心六：板块轮动预测
    # ============================================================
    def predict_sector_rotation(self, sector_data: pd.DataFrame,
                                history_kline: dict = None) -> Dict[str, Any]:
        """基于多因子预测板块轮动方向"""
        if sector_data.empty:
            return {'hot_sectors': [], 'cold_sectors': [], 'rotation_signal': '数据不足'}

        analysis = {
            'hot_sectors': [],
            'cold_sectors': [],
            'rotation_signal': '',
            'next_opportunity': [],
            'details': []
        }

        for _, row in sector_data.iterrows():
            sector = row.get('板块', '')
            change = row.get('今日涨幅%', 0)
            flow = row.get('主力净流入(亿)', 0)
            low_signal = str(row.get('低位信号', ''))

            if change > 2 and flow > 10:
                analysis['hot_sectors'].append({
                    'name': sector,
                    'change': change,
                    'flow': flow,
                    'reason': '涨幅+资金共振'
                })

            if change < -1 or flow < -5:
                analysis['cold_sectors'].append({
                    'name': sector,
                    'change': change,
                    'flow': flow
                })

            if '强' in low_signal and flow > 0:
                analysis['next_opportunity'].append({
                    'name': sector,
                    'reason': '低位+资金流入，轮动起飞概率高'
                })

        if len(analysis['hot_sectors']) >= 2:
            analysis['rotation_signal'] = '强势轮动中，跟随主线'
        elif len(analysis['next_opportunity']) >= 1:
            analysis['rotation_signal'] = '低位股/板块轮动即将启动'
        elif len(analysis['cold_sectors']) >= 3:
            analysis['rotation_signal'] = '市场偏弱，防御为主'
        else:
            analysis['rotation_signal'] = '震荡格局，精选个股'

        return analysis

    # ============================================================
    #  核心七：组合配置建议
    # ============================================================
    def portfolio_advice(self, holdings: List[Dict], total_capital: float,
                         market_signals: Dict = None) -> Dict[str, Any]:
        """基于多因子给出组合配置建议"""
        advice = {
            'total_capital': total_capital,
            'current_allocation': {},
            'suggested_allocation': {
                'core': {'pct': 60, 'desc': '核心仓位：主线强势板块龙头'},
                'satellite': {'pct': 30, 'desc': '卫星仓位：轮动机会+弹性品种'},
                'defensive': {'pct': 10, 'desc': '防御仓位：黄金/债券/ETF'}
            },
            'actions': [],
            'warnings': []
        }

        total_position = 0
        sector_weights = {}
        for h in holdings:
            pct = h.get('position', 0) / total_capital * 100
            total_position += pct
            sector = h.get('sector', '未知')
            sector_weights[sector] = sector_weights.get(sector, 0) + pct
            advice['current_allocation'][h['code']] = round(pct, 1)

        # 仓位检查
        if total_position > 80:
            advice['warnings'].append(f'总仓位{total_position:.0f}%偏高，建议控制在70%以内')
        elif total_position < 30:
            advice['warnings'].append(f'总仓位{total_position:.0f}%偏低，可适度建仓')

        # 行业集中度
        for sector, wt in sector_weights.items():
            if wt > 40:
                advice['warnings'].append(f'{sector}板块占比{wt:.0f}%过高，建议分散至<30%')

        # 个股集中度检查
        for h in holdings:
            pct = h.get('position', 0) / total_capital * 100
            if pct > 30:
                advice['warnings'].append(f'{h["code"]}占比{pct:.0f}%过高，建议<30%')

        # 现金流
        cash_pct = 100 - total_position
        advice['cash_pct'] = round(cash_pct, 1)
        if cash_pct > 50:
            advice['actions'].append('现金占比高，建议分批建仓主线板块')
        elif cash_pct > 30:
            advice['actions'].append('现金充裕，可寻找低位轮动机会入场')
        elif cash_pct < 10:
            advice['actions'].append('现金较少，保留子弹等待更好机会，不要满仓')
        else:
            advice['actions'].append('仓位合理，关注轮动机会')

        return advice

    # ============================================================
    #  核心八：市场情绪判断
    # ============================================================
    def market_sentiment(self, df: pd.DataFrame, limit_up_count: int = 0,
                         limit_down_count: int = 0,
                         up_down_ratio: float = 1.0) -> Dict[str, Any]:
        """判断当前市场情绪/风格"""
        sentiment = {
            'overall': '中性',
            'market_style': '均衡',
            'risk_appetite': '中性',
            'details': []
        }

        trending_count = 0
        if 'MA_BULL' in df.columns:
            trending_count += int(df.iloc[-1].get('MA_BULL', 0))
        if 'ADX_TRENDING' in df.columns:
            trending_count += int(df.iloc[-1].get('ADX_TRENDING', 0))

        if trending_count >= 2:
            sentiment['overall'] = '偏多'
            sentiment['details'].append('趋势指标多头')
        elif trending_count == 0:
            sentiment['overall'] = '偏空'
            sentiment['details'].append('趋势指标空头')

        if limit_up_count > 80:
            sentiment['risk_appetite'] = '高（风险偏好）'
            sentiment['details'].append(f'涨停{limit_up_count}家，市场情绪高')
        elif limit_up_count > 40:
            sentiment['risk_appetite'] = '中等'
            sentiment['details'].append(f'涨停{limit_up_count}家，情绪正常')
        else:
            sentiment['risk_appetite'] = '低（防御）'
            sentiment['details'].append(f'涨停{limit_up_count}家，情绪低迷')

        if limit_down_count > 30:
            sentiment['details'].append(f'跌停{limit_down_count}家，⚠️注意风险')
            sentiment['overall'] = '偏空'

        if up_down_ratio > 2:
            sentiment['market_style'] = '强势普涨'
        elif up_down_ratio > 1.2:
            sentiment['market_style'] = '偏多震荡'
        elif up_down_ratio < 0.5:
            sentiment['market_style'] = '弱势普跌'
        elif up_down_ratio < 0.8:
            sentiment['market_style'] = '偏空震荡'
        else:
            sentiment['market_style'] = '均衡震荡'

        return sentiment

    # ============================================================
    #  LLM增强（可选）
    # ============================================================
    def enhance_with_llm(self, base_analysis: Dict, prompt_template: str = None) -> str:
        """使用LLM增强分析，提供自然语言解读"""
        if not self.use_llm or not self.api_key:
            return "（LLM未启用或API Key未配置）"

        try:
            prompt = f"""
你是一位资深A股分析师。请根据以下技术分析数据，用中文给出简明扼要的操作建议（200字内）。

分析数据：
{base_analysis}

请从以下角度回答：
1. 当前该股的总体技术状态（多头/空头/震荡）
2. 近期买卖点的概率（仅给出方向性判断）
3. 风险提示
4. 一句话操作建议

注意：必须在开头声明"以下分析仅供参考，不构成投资建议"。
"""
            if self.llm_provider == "openai" and HAS_OPENAI:
                response = openai.ChatCompletion.create(
                    model="gpt-4o-mini",
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=500
                )
                return response.choices[0].message.content

            elif self.llm_provider == "gemini" and HAS_GEMINI:
                model = genai.GenerativeModel('gemini-2.0-flash')
                response = model.generate_content(prompt)
                return response.text

            return "（LLM调用失败）"

        except Exception as e:
            return f"（LLM调用异常: {str(e)}）"
