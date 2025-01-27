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
        dict: 处理后的数据
    """
    processed_data = {}
    for ticker, data in raw_data.items():
        # 数据已经是Series格式，直接处理
        processed_data[ticker] = data.dropna()
    return processed_data
