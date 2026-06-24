#!/usr/bin/env python3
"""
A股智选 - 一键擒龙：全市场扫描引擎
多线程并发获取K线 + AI评审，筛选出买入信号的股票和行业
"""

import pandas as pd
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Dict, List, Optional, Callable
import hashlib

from data_provider import get_kline, get_fund_flow
from ai_engine import AIEngine
from technical_indicators import add_indicators_streamlined, TechnicalIndicators

# ==================== 预置股票池 ====================
STOCK_POOLS = {
    "长鑫存储概念": [
        ("688126", "沪硅产业/硅片"), ("688037", "芯源微/Track"),
        ("688012", "中微公司/刻蚀"), ("002371", "北方华创/PVD"),
        ("688072", "拓荆科技/CVD"), ("688147", "微导纳米/ALD"),
        ("600641", "万业企业/离子注入"), ("688120", "华海清科/CMP"),
        ("688082", "盛美上海/清洗"), ("603690", "至纯科技/高纯"),
        ("688361", "中科飞测/检测"), ("300567", "精测电子/检测"),
        ("688200", "华峰测控/测试"), ("300604", "长川科技/测试"),
        ("002156", "通富微电/封测"), ("600584", "长电科技/封测"),
        ("000021", "深科技/封测"),
        ("002409", "雅克科技/前驱体"), ("688019", "安集科技/CMP液"),
        ("300054", "鼎龙股份/CMP垫"), ("300236", "上海新阳/电镀液"),
        ("300666", "江丰电子/靶材"), ("300346", "南大光电/光胶"),
        ("603650", "彤程新材/光胶"), ("300655", "晶瑞电材/电化品"),
        ("688596", "正帆科技/气柜"),
        ("603986", "兆易创新/DRAM"), ("688525", "佰维存储/模组"),
    ],
    "半导体设备": [
        ("688017", "绿的谐波"), ("688012", "中微公司"), ("002371", "北方华创"),
        ("688082", "盛美上海"), ("688200", "华峰测控"), ("603690", "至纯科技"),
        ("688596", "正帆科技"), ("300316", "晶盛机电"),
    ],
    "存储芯片": [
        ("688981", "中芯国际/存储"), ("002049", "紫光国微"), ("603986", "兆易创新"),
        ("688525", "佰维存储"), ("300672", "国科微"), ("688110", "东芯股份"),
        ("002415", "海康威视"),
    ],
    "AI算力": [
        ("300502", "新易盛/光模块"), ("688256", "寒武纪/AI芯片"), ("000977", "浪潮信息"),
        ("603019", "中科曙光"), ("688041", "海光信息"), ("300308", "中际旭创"),
        ("002230", "科大讯飞"), ("688111", "金山办公"),
    ],
    "CPO光通信": [
        ("300308", "中际旭创/龙头"), ("300502", "新易盛"),
        ("300394", "天孚通信"), ("688498", "源杰科技"),
        ("300620", "光库科技"), ("300570", "太辰光"),
        ("688195", "腾景科技"), ("300548", "博创科技"),
        ("688313", "仕佳光子"), ("002281", "光迅科技"),
        ("603083", "剑桥科技"), ("600487", "亨通光电"),
        ("000988", "华工科技"), ("002902", "铭普光磁"),
        ("688205", "德科立"), ("300757", "罗博特科"),
        ("301205", "联特科技"), ("300913", "兆龙互连"),
        ("688662", "富信科技"), ("301191", "菲菱科思"),
        ("300252", "金信诺"), ("002396", "星网锐捷"),
        ("603118", "共进股份"), ("301183", "东田微"),
    ],
    "新能源/光伏": [
        ("300750", "宁德时代"), ("601012", "隆基绿能"), ("002459", "晶澳科技"),
        ("688599", "天合光能"), ("300274", "阳光电源"), ("300014", "亿纬锂能"),
        ("002812", "恩捷股份"), ("300763", "锦浪科技"),
    ],
    "军工": [
        ("600760", "中航沈飞"), ("600893", "航发动力"), ("002025", "航天电器"),
        ("600765", "中航重机"), ("002013", "中航机电"),
    ],
    "消费品/白酒": [
        ("600519", "贵州茅台"), ("000858", "五粮液"), ("000568", "泸州老窖"),
        ("603288", "海天味业"), ("002304", "洋河股份"),
    ],
    "金融/银行": [
        ("000001", "平安银行"), ("600036", "招商银行"), ("601318", "中国平安"),
        ("002142", "宁波银行"), ("600030", "中信证券"), ("300059", "东方财富"),
    ],
    "医药": [
        ("300760", "迈瑞医疗"), ("600276", "恒瑞医药"), ("603259", "药明康德"),
        ("300015", "爱尔眼科"), ("000661", "长春高新"),
    ],
    "5G/通信": [
        ("300394", "天孚通信"), ("300136", "信维通信"), ("600487", "亨通光电"),
        ("002475", "立讯精密"), ("000063", "中兴通讯"),
    ],
    "MLCC概念": [
        ("000636", "风华高科/MLCC"), ("300408", "三环集团/MLCC"),
        ("603678", "火炬电子/特容"), ("300726", "宏达电子/钽容"),
        ("603267", "鸿远电子/MLCC"), ("002859", "洁美科技/载带"),
        ("000733", "振华科技/元器件"), ("301566", "达利凯普/射频MLCC"),
        ("605376", "博迁新材/镍粉"), ("300285", "国瓷材料/瓷粉"),
    ],
    "先进封装(Chiplet)": [
        ("002156", "通富微电/封测"), ("600584", "长电科技/封测"),
        ("002185", "华天科技/封测"), ("603005", "晶方科技/WLCSP"),
        ("688362", "甬矽电子/先进封测"), ("002436", "兴森科技/IC载板"),
        ("002916", "深南电路/封装基板"), ("688981", "中芯国际/制造"),
    ],
    "玻璃基板": [
        ("603773", "沃格光电/玻璃基板"), ("600707", "彩虹股份/基板玻璃"),
        ("600552", "凯盛科技/玻璃基材"), ("688327", "蓝特光学/光学玻璃"),
        ("300256", "星星科技/玻璃盖板"),
    ],
    "氮化镓GaN": [
        ("600460", "士兰微/GaN器件"), ("600703", "三安光电/GaN外延"),
        ("688396", "华润微/GaN功率"), ("600745", "闻泰科技/GaN快充"),
        ("300456", "赛微电子/GaN代工"), ("300708", "聚灿光电/GaN外延"),
    ],
    "碳化硅SiC": [
        ("688234", "天岳先进/衬底"), ("002617", "露笑科技/衬底"),
        ("603290", "斯达半导/模组"), ("688187", "时代电气/器件"),
        ("300316", "晶盛机电/长晶"), ("003031", "中瓷电子/陶瓷"),
    ],
    "磷化铟InP": [
        ("002023", "海特高新/InP外延"), ("300102", "乾照光电/化合物"),
        ("002281", "光迅科技/InP光芯"), ("688313", "仕佳光子/光芯片"),
        ("688048", "长光华芯/激光芯"), ("002222", "福晶科技/晶体"),
    ],
    "人造钻石": [
        ("301071", "力量钻石/金刚石"), ("600172", "黄河旋风/超硬"),
        ("000519", "中兵红箭/超硬"), ("300179", "四方达/金刚石"),
        ("688028", "沃尔德/金刚石"),
    ],
}


def _build_strategy(signals: dict, levels: dict, direction: str) -> str:
    """根据信号和价位生成简洁操作策略"""
    if direction == 'buy':
        parts = [f"买入{levels['buy_point_aggressive']}"]
        if levels.get('buy_point_conservative'):
            parts.append(f"保守{levels['buy_point_conservative']}")
        if signals['buy_score'] >= 40:
            parts.insert(0, "强推")
        elif signals['buy_score'] >= 20:
            parts.insert(0, "推荐")
        elif signals['buy_score'] >= 10:
            parts.insert(0, "关注")
        return '/'.join(parts)
    else:
        parts = []
        tp1 = levels.get('take_profit_1')
        tp2 = levels.get('take_profit_2')
        sl = levels.get('stop_loss_normal')
        if tp1:
            parts.append(f"止盈{tp1}(+{levels.get('tp1_pct','')}%)")
        if tp2 and tp2 > tp1:
            parts.append(f"目标{tp2}")
        if sl:
            parts.append(f"止损{sl}(-{levels.get('sl_pct','')}%)")
        return ' | '.join(parts) if parts else '待定'


def _scan_single_stock(code: str, name: str, sector: str, days: int = 90,
                       progress_callback: Callable = None) -> Optional[Dict]:
    """扫描单只股票"""
    try:
        df = get_kline(code, days=days)
        if df.empty or len(df) < 30:
            return None

        df = add_indicators_streamlined(df)
        engine = AIEngine()
        signals = engine.detect_signals(df)
        risk = engine.calculate_risk_score(df, signals)
        levels = engine.calculate_trade_levels(df)
        fund = get_fund_flow(code)

        latest = df.iloc[-1]

        # 优先使用腾讯实时行情覆盖日K线的昨日数据
        realtime_price = fund.get('最新价', 0)
        realtime_change = fund.get('涨跌幅', None)
        if realtime_price and realtime_price > 0:
            display_price = round(float(realtime_price), 2)
        else:
            display_price = round(latest['close'], 2)
        if realtime_change is not None and realtime_change != 0:
            display_change = round(float(realtime_change), 2)
        else:
            display_change = round(latest.get('Change', 0), 2)

        # === 信号质量回测：基于实际K线，而非策略模拟 ===
        # 20日/60日涨幅（买入持有基准）
        if len(df) >= 20:
            ret_20d = (df['close'].iloc[-1] / df['close'].iloc[-20] - 1) * 100
        else:
            ret_20d = 0
        if len(df) >= 60:
            ret_60d = (df['close'].iloc[-1] / df['close'].iloc[-60] - 1) * 100
        else:
            ret_60d = 0

        # MACD金叉信号质量：统计历史上每次金叉后20日的平均收益
        macd_cross_count = int(df.get('MACD_GOLDEN', pd.Series([0]*len(df))).sum())
        cross_returns = []
        if 'MACD_GOLDEN' in df.columns and macd_cross_count > 0:
            cross_idx = df.index[df['MACD_GOLDEN'] == 1]
            for i, idx in enumerate(cross_idx):
                pos = df.index.get_loc(idx)
                if pos + 20 < len(df):
                    fwd_ret = (df['close'].iloc[pos+20] / df['close'].iloc[pos] - 1) * 100
                    cross_returns.append(fwd_ret)
        avg_cross_return = round(np.mean(cross_returns), 2) if cross_returns else 0
        cross_win_rate = round(sum(1 for r in cross_returns if r > 0) / max(len(cross_returns), 1) * 100, 1)

        # 均线多头占比（趋势强度）
        bull_days = int(df.get('MA_BULL', pd.Series([0]*len(df))).sum())
        total_days = len(df)
        bull_ratio = round(bull_days / max(total_days, 1) * 100, 1)

        # KDJ金叉信号质量
        kdj_cross_count = int(df.get('KDJ_GOLDEN', pd.Series([0]*len(df))).sum())
        kdj_cross_returns = []
        if 'KDJ_GOLDEN' in df.columns and kdj_cross_count > 0:
            kdj_cross_idx = df.index[df['KDJ_GOLDEN'] == 1]
            for idx in kdj_cross_idx:
                pos = df.index.get_loc(idx)
                if pos + 20 < len(df):
                    fwd_ret = (df['close'].iloc[pos+20] / df['close'].iloc[pos] - 1) * 100
                    kdj_cross_returns.append(fwd_ret)
        avg_kdj_return = round(np.mean(kdj_cross_returns), 2) if kdj_cross_returns else 0
        kdj_win_rate = round(sum(1 for r in kdj_cross_returns if r > 0) / max(len(kdj_cross_returns), 1) * 100, 1)

        result = {
            'code': code,
            'name': name,
            'sector': sector,
            'price': display_price,
            'change%': display_change,
            'signal': signals['overall'],
            'buy_score': signals['buy_score'],
            'sell_score': signals['sell_score'],
            'rsi': round(latest.get('RSI', 50), 1),
            'macd_status': '多头' if latest.get('MACD', 0) > latest.get('Signal', 0) else '空头',
            'ma_status': '多头排列' if latest.get('MA_BULL', 0) == 1 else ('空头排列' if latest.get('MA_BEAR', 0) == 1 else '震荡'),
            'vol_ratio': round(latest.get('VOLUME_RATIO', 1), 1),
            'atr_pct': round(latest.get('ATR_PCT', 0), 1),
            'risk_score': risk['score'],
            'risk_level': risk['level'],
            'entry_price': levels['buy_point_aggressive'],
            'entry_conservative': levels['buy_point_conservative'],
            'stop_loss': levels['stop_loss_normal'],
            'stop_loss_tight': levels['stop_loss_tight'],
            'take_profit_1': levels['take_profit_1'],
            'take_profit_2': levels['take_profit_2'],
            'tp1_pct': round((levels['take_profit_1'] / latest['close'] - 1) * 100, 1),
            'tp2_pct': round((levels['take_profit_2'] / latest['close'] - 1) * 100, 1),
            'sl_pct': round((1 - levels['stop_loss_normal'] / latest['close']) * 100, 1),
            'buy_strategy': _build_strategy(signals, levels, 'buy'),
            'sell_strategy': _build_strategy(signals, levels, 'sell'),
            'buy_reasons': '; '.join(signals['buy_signals']) if signals['buy_signals'] else '',
            'sell_reasons': '; '.join(signals['sell_signals']) if signals['sell_signals'] else '',
            'fund_trend': fund.get('趋势', 'N/A'),
            'fund_flow': fund.get('主力净流入(亿)', 0),
            'pe': fund.get('市盈率', 0),
            'ret_20d': round(ret_20d, 2),
            'ret_60d': round(ret_60d, 2),
            'macd_cross_n': macd_cross_count,
            'macd_cross_ret': avg_cross_return,
            'macd_cross_wr': cross_win_rate,
            'kdj_cross_n': kdj_cross_count,
            'kdj_cross_ret': avg_kdj_return,
            'kdj_cross_wr': kdj_win_rate,
            'bull_ratio': bull_ratio,
        }

        if progress_callback:
            progress_callback(code, name)

        return result

    except Exception as e:
        return None


def scan_stocks(stock_list: List[tuple], days: int = 90,
                max_workers: int = 5, progress_callback: Callable = None) -> pd.DataFrame:
    """
    并发扫描股票列表
    """
    results = []
    total = len(stock_list)
    done_count = [0]

    def _on_stock_done(future):
        try:
            result = future.result()
            if result:
                results.append(result)
        except Exception:
            pass
        done_count[0] += 1
        if progress_callback:
            progress_callback(None, None, done_count[0], total)

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = []
        for code, name, sector in stock_list:
            future = executor.submit(_scan_single_stock, code, name, sector, days)
            future.add_done_callback(_on_stock_done)
            futures.append(future)

        for future in futures:
            future.result()

    if not results:
        return pd.DataFrame()

    df = pd.DataFrame(results)
    df = df.sort_values('buy_score', ascending=False)

    return df


def scan_pool(pool_name: str, days: int = 90, max_workers: int = 5,
              progress_callback: Callable = None) -> pd.DataFrame:
    """扫描预置股票池"""
    if pool_name not in STOCK_POOLS:
        return pd.DataFrame()

    stock_list = [(code, name, pool_name) for code, name in STOCK_POOLS[pool_name]]
    return scan_stocks(stock_list, days=days, max_workers=max_workers,
                       progress_callback=progress_callback)


def scan_all_pools(days: int = 90, max_workers: int = 5,
                   progress_callback: Callable = None) -> pd.DataFrame:
    """扫描全部预置股票池"""
    all_stocks = []
    for sector, stocks in STOCK_POOLS.items():
        for code, name in stocks:
            all_stocks.append((code, name, sector))

    return scan_stocks(all_stocks, days=days, max_workers=max_workers,
                       progress_callback=progress_callback)


def filter_buy_signals(df: pd.DataFrame,
                       min_buy_score: int = 10) -> pd.DataFrame:
    """筛选有买入信号的股票"""
    if df.empty:
        return df
    buy_signals = ['STRONG_BUY', 'BUY', 'WEAK_BUY']
    return df[df['signal'].isin(buy_signals) & (df['buy_score'] >= min_buy_score)]


def summarize_by_sector(df: pd.DataFrame) -> pd.DataFrame:
    """按行业汇总扫描结果"""
    if df.empty:
        return pd.DataFrame()

    buy_signals = ['STRONG_BUY', 'BUY', 'WEAK_BUY']

    summary = df.groupby('sector').agg(
        股票数量=('code', 'count'),
        买入信号数=('signal', lambda x: sum(1 for s in x if s in buy_signals)),
        平均买入分=('buy_score', 'mean'),
        最大买入分=('buy_score', 'max'),
        平均涨幅=('change%', 'mean'),
    ).reset_index()

    summary['平均买入分'] = summary['平均买入分'].round(1)
    summary['平均涨幅'] = summary['平均涨幅'].round(2)

    return summary.sort_values('买入信号数', ascending=False)


if __name__ == '__main__':
    print("=== 一键擒龙 测试 ===")
    test_stocks = [
        ("000001", "平安银行", "银行"),
        ("600519", "贵州茅台", "白酒"),
        ("300750", "宁德时代", "新能源"),
    ]

    def progress(code, name, done, total):
        print(f"  [{done}/{total}] {code} {name}")

    df = scan_stocks(test_stocks, days=60, max_workers=3, progress_callback=progress)
    if not df.empty:
        print(f"\n扫描结果 ({len(df)}只):")
        print(df[['code', 'name', 'sector', 'price', 'signal',
                   'buy_score', 'rsi', 'risk_level']].to_string())
