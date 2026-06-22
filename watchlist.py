#!/usr/bin/env python3
"""
A股智选 - 自选股票池 / 关注列表
持久化存储，跨会话保留
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

WATCHLIST_FILE = os.path.join(os.path.dirname(__file__), 'data', 'watchlist.json')
os.makedirs(os.path.dirname(WATCHLIST_FILE), exist_ok=True)


def load_watchlist() -> List[Dict]:
    """加载关注列表"""
    if not os.path.exists(WATCHLIST_FILE):
        return []
    try:
        with open(WATCHLIST_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data.get('stocks', [])
    except Exception:
        return []


def save_watchlist(stocks: List[Dict]):
    """保存关注列表"""
    try:
        with open(WATCHLIST_FILE, 'w', encoding='utf-8') as f:
            json.dump({'stocks': stocks, 'updated_at': datetime.now().isoformat()},
                      f, ensure_ascii=False, indent=2)
    except Exception:
        pass


def add_to_watchlist(code: str, name: str = '', sector: str = '',
                     price: float = 0, notes: str = '') -> bool:
    """添加股票到关注列表，已存在则跳过"""
    stocks = load_watchlist()
    existing = [s for s in stocks if s['code'] == code]
    if existing:
        existing[0]['updated_at'] = datetime.now().isoformat()
        if name:
            existing[0]['name'] = name
        if sector:
            existing[0]['sector'] = sector
    else:
        stocks.append({
            'code': code,
            'name': name,
            'sector': sector if sector else '用户关注',
            'price': price,
            'notes': notes,
            'added_at': datetime.now().isoformat(),
            'updated_at': datetime.now().isoformat(),
        })
    save_watchlist(stocks)
    return True


def remove_from_watchlist(code: str) -> bool:
    """从关注列表移除"""
    stocks = load_watchlist()
    before = len(stocks)
    stocks = [s for s in stocks if s['code'] != code]
    if len(stocks) < before:
        save_watchlist(stocks)
        return True
    return False


def is_in_watchlist(code: str) -> bool:
    """检查是否已在关注列表"""
    stocks = load_watchlist()
    return any(s['code'] == code for s in stocks)


def get_watchlist_pool() -> tuple:
    """获取关注列表，返回 (pool_name, [(code, name, sector)]) 格式"""
    stocks = load_watchlist()
    if not stocks:
        return ('我的关注', [])
    pool = [(s['code'], s.get('name', ''), s.get('sector', '用户关注')) for s in stocks]
    return ('我的关注', pool)


def get_watchlist_df():
    """获取关注列表DataFrame"""
    import pandas as pd
    stocks = load_watchlist()
    if not stocks:
        return pd.DataFrame()
    return pd.DataFrame(stocks)


def clear_watchlist():
    """清空关注列表"""
    save_watchlist([])


def get_watchlist_count() -> int:
    return len(load_watchlist())
