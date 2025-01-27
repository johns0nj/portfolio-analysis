# data_processing.py
import pandas as pd

"""
Module for processing and cleaning raw stock market data.

This module contains functions for data cleaning, normalization,
and preparation for portfolio analysis.
"""

def process_stock_data(raw_data):
    """
    处理原始股票数据

    Args:
        raw_data (dict): 包含股票数据的字典

    Returns:
        dict: 处理后的数据，只包含调整后收盘价
    """
    processed_data = {}
    for ticker, data in raw_data.items():
        # 确保数据包含 'Adj Close' 列
        if 'Adj Close' not in data.columns:
            # 如果没有 'Adj Close'，使用 'Close' 列
            processed_data[ticker] = data['Close'].copy()
        else:
            processed_data[ticker] = data['Adj Close'].copy()
            
        # 删除缺失值
        processed_data[ticker] = processed_data[ticker].dropna()
        
    return processed_data
