# calculations.py
import pandas as pd
import numpy as np
from data_fetching import fetch_stock_data
import logging

"""
Module for performing portfolio analysis calculations.

This module provides functions for calculating various portfolio metrics,
including returns, volatility, Sharpe ratio, and Monte Carlo simulations.
"""

def calculate_portfolio_returns(processed_data, portfolio_weights):
    """
    计算投资组合收益率
    
    Args:
        processed_data (dict): 已处理的股票数据
        portfolio_weights (dict): 投资组合权重
        
    Returns:
        pd.Series: 投资组合的收益率序列
    """
    price_data = pd.DataFrame()
    
    # 直接使用处理后的数据，因为它已经只包含了我们需要的价格数据
    for ticker, data in processed_data.items():
        price_data[ticker] = data  # 不再需要访问 'Adj Close'
        
    # 计算收益率
    returns = price_data.pct_change()
    
    # 计算投资组合收益率
    portfolio_returns = returns.dot(pd.Series(portfolio_weights))
    
    return portfolio_returns

def calculate_portfolio_volatility(portfolio_returns):
    """
    Calculates the annualized volatility of portfolio returns.

    Args:
        portfolio_returns (pandas.Series): Daily portfolio returns.

    Returns:
        float: Annualized portfolio volatility (standard deviation).

    Notes:
        Annualization assumes 252 trading days per year.
    """
    daily_volatility = portfolio_returns.std()
    annualized_volatility = daily_volatility * np.sqrt(252)
    return annualized_volatility

def calculate_sharpe_ratio(portfolio_returns, risk_free_rate=0.01):
    """
    Calculates the annualized Sharpe ratio for the portfolio.

    Args:
        portfolio_returns (pandas.Series): Daily portfolio returns.
        risk_free_rate (float, optional): Annual risk-free rate. Defaults to 0.01 (1%).

    Returns:
        float: Annualized Sharpe ratio.

    Notes:
        - Sharpe ratio is calculated as (portfolio_return - risk_free_rate) / portfolio_volatility
        - Annualization assumes 252 trading days per year
    """
    daily_risk_free_rate = risk_free_rate / 252
    excess_returns = portfolio_returns - daily_risk_free_rate
    sharpe_ratio = excess_returns.mean() / excess_returns.std()
    annualized_sharpe_ratio = sharpe_ratio * np.sqrt(252)
    return annualized_sharpe_ratio

def calculate_beta(portfolio_returns, benchmark_returns):
    """
    Calculates the portfolio's beta relative to a benchmark.

    Args:
        portfolio_returns (pandas.Series): Daily portfolio returns.
        benchmark_returns (pandas.Series): Daily benchmark returns.

    Returns:
        float: Portfolio beta coefficient.

    Notes:
        Beta measures the portfolio's systematic risk relative to the market,
        where beta = 1 indicates same volatility as the market.
    """
    covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
    benchmark_variance = benchmark_returns.var()
    beta = covariance / benchmark_variance
    return beta

def calculate_var(portfolio_returns, confidence_level=0.05):
    """
    Calculates the Value at Risk (VaR) for the portfolio.

    Args:
        portfolio_returns (pandas.Series): Daily portfolio returns.
        confidence_level (float, optional): Confidence level for VaR calculation.
            Defaults to 0.05 (95% confidence).

    Returns:
        float: Value at Risk at specified confidence level.

    Notes:
        VaR represents the maximum expected loss at the given confidence level.
    """
    var = np.percentile(portfolio_returns, 100 * confidence_level)
    return var

def monte_carlo_simulation(processed_data, portfolio_weights, num_simulations=1000, forecast_days=252):
    """
    进行蒙特卡洛模拟
    """
    logger = logging.getLogger('portfolio_analyzer')
    
    try:
        logger.info(f"开始蒙特卡洛模拟，模拟次数：{num_simulations}")
        weights = np.array(list(portfolio_weights.values()))
        
        # 直接使用处理后的数据
        price_data = pd.DataFrame(processed_data)
        
        daily_returns = price_data.pct_change().dropna()
        mean_returns = daily_returns.mean()
        cov_matrix = daily_returns.cov()
        
        simulation_results = np.zeros(num_simulations)
        for i in range(num_simulations):
            if i % 200 == 0:
                logger.debug(f"完成 {i} 次模拟")
                
            simulated_returns = np.random.multivariate_normal(mean_returns, cov_matrix, forecast_days)
            portfolio_simulated_returns = np.cumprod(np.dot(simulated_returns, weights) + 1)
            simulation_results[i] = portfolio_simulated_returns[-1]
        
        logger.info("蒙特卡洛模拟完成")
        return simulation_results
        
    except Exception as e:
        logger.error(f"蒙特卡洛模拟出错: {str(e)}")
        raise

def calculate_portfolio_metrics(processed_data, portfolio_weights):
    """
    计算投资组合的关键指标
    """
    metrics = {}
    portfolio_returns = calculate_portfolio_returns(processed_data, portfolio_weights)
    metrics['Expected Return'] = portfolio_returns.mean() * 252
    metrics['Volatility'] = calculate_portfolio_volatility(portfolio_returns)
    metrics['Sharpe Ratio'] = calculate_sharpe_ratio(portfolio_returns)
    
    # 获取基准数据并处理
    try:
        benchmark_data = fetch_stock_data(['^GSPC'], portfolio_returns.index[0], portfolio_returns.index[-1])
        benchmark_returns = benchmark_data['^GSPC'].pct_change().dropna()
        
        # 对齐日期
        aligned_returns = pd.concat([portfolio_returns, benchmark_returns], axis=1).dropna()
        portfolio_returns_aligned = aligned_returns.iloc[:, 0]
        benchmark_returns_aligned = aligned_returns.iloc[:, 1]
        
        metrics['Beta'] = calculate_beta(portfolio_returns_aligned, benchmark_returns_aligned)
    except Exception as e:
        metrics['Beta'] = None
        logging.warning(f"无法计算 Beta: {str(e)}")
    
    metrics['VaR'] = calculate_var(portfolio_returns)
    
    return metrics
 