# data_fetching.py
import yfinance as yf
import logging

"""
Module for fetching stock data from various financial data sources.

This module provides functionality to retrieve historical stock data
using the yfinance API.
"""

def fetch_stock_data(ticker_list, start_date, end_date):
    """
    获取股票数据
    
    Args:
        ticker_list (list): 股票代码列表
        start_date (str): 开始日期
        end_date (str): 结束日期
        
    Returns:
        dict: 包含股票数据的字典
    """
    data = {}
    for ticker in ticker_list:
        try:
            stock = yf.download(ticker, start=start_date, end=end_date)
            if 'Adj Close' in stock.columns:
                data[ticker] = stock['Adj Close']
            else:
                data[ticker] = stock['Close']
        except Exception as e:
            logging.warning(f"获取 {ticker} 数据时出错: {str(e)}")
            continue
    return data
