# visualization.py
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import logging
from matplotlib.widgets import CheckButtons

"""
Module for creating portfolio analysis visualizations.

This module provides functions for plotting various aspects of portfolio
performance, including returns, efficient frontier, and Monte Carlo simulations.
"""

def plot_portfolio_performance(portfolio_returns, benchmark_returns=None):
    """
    绘制投资组合表现，包括与基准的对比
    
    Args:
        portfolio_returns (pd.Series): 投资组合收益率
        benchmark_returns (pd.Series, optional): 基准收益率
    """
    # 创建图表和布局
    fig = plt.figure(figsize=(15, 8))  # 加宽图表
    
    # 创建主图表，并为右侧留出空间
    ax = plt.axes([0.1, 0.1, 0.7, 0.8])  # [left, bottom, width, height]
    
    # 计算累积收益
    portfolio_cumulative = (1 + portfolio_returns).cumprod()
    portfolio_line, = ax.plot(portfolio_cumulative.index, portfolio_cumulative.values, 
                            label='Portfolio', linewidth=2)
    
    benchmark_line = None
    if benchmark_returns is not None:
        benchmark_cumulative = (1 + benchmark_returns).cumprod()
        benchmark_line, = ax.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
                                label='S&P 500', linewidth=2, linestyle='--')
    
    # 设置主图表属性
    ax.set_title('Portfolio Performance vs Benchmark')
    ax.set_xlabel('Date')
    ax.set_ylabel('Cumulative Return')
    ax.grid(True)
    ax.legend()
    
    # 创建多选框数据
    lines = [portfolio_line]
    labels = ['Portfolio']
    visibility = [True]
    
    if benchmark_line is not None:
        lines.append(benchmark_line)
        labels.append('S&P 500')
        visibility.append(True)
    
    # 创建多选框，放在右侧空白处
    rax = plt.axes([0.85, 0.4, 0.1, 0.2])  # [left, bottom, width, height]
    rax.set_frame_on(False)  # 移除边框
    rax.set_xticks([])  # 移除刻度
    rax.set_yticks([])
    
    # 创建多选框
    check = CheckButtons(rax, labels, visibility)
    
    # 自定义多选框样式
    try:
        for rect in check.rectangles:
            rect.set_facecolor('lightgray')
            rect.set_alpha(0.3)
    except AttributeError:
        try:
            for rect in check.boxes:
                rect.set_facecolor('lightgray')
                rect.set_alpha(0.3)
        except AttributeError:
            pass
    
    # 定义多选框回调函数
    def func(label):
        index = labels.index(label)
        lines[index].set_visible(not lines[index].get_visible())
        plt.draw()
    
    check.on_clicked(func)
    
    # 保存图片前确保所有元素都可见
    for line in lines:
        line.set_visible(True)
    
    # 保存和显示
    plt.savefig('portfolio_performance.png', bbox_inches='tight')
    plt.show()
    plt.close()

def plot_efficient_frontier(processed_data, num_portfolios=5000, risk_free_rate=0.01):
    """
    绘制有效前沿
    """
    logger = logging.getLogger('portfolio_analyzer')
    
    try:
        logger.debug(f"生成有效前沿，模拟 {num_portfolios} 个组合")
        price_data = pd.DataFrame()
        for ticker, data in processed_data.items():
            # 直接使用处理后的数据
            price_data[ticker] = data
        
        daily_returns = price_data.pct_change().dropna()
        mean_returns = daily_returns.mean()
        cov_matrix = daily_returns.cov()
        num_assets = len(processed_data.keys())
        
        # Create lists instead of numpy arrays for collecting results
        returns = []
        volatilities = []
        sharpe_ratios = []
        
        for i in range(num_portfolios):
            weights = np.random.random(num_assets)
            weights /= np.sum(weights)
            
            portfolio_return = np.sum(mean_returns * weights) * 252
            portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix * 252, weights)))
            sharpe_ratio = (portfolio_return - risk_free_rate) / portfolio_volatility
            
            returns.append(portfolio_return)
            volatilities.append(portfolio_volatility)
            sharpe_ratios.append(sharpe_ratio)
        
        # Convert to numpy arrays for plotting
        returns = np.array(returns)
        volatilities = np.array(volatilities)
        sharpe_ratios = np.array(sharpe_ratios)
        
        plt.figure(figsize=(10, 7))
        plt.scatter(volatilities, returns, c=sharpe_ratios, cmap='viridis', marker='o', s=10, alpha=0.3)
        plt.colorbar(label='Sharpe Ratio')
        plt.xlabel('Volatility (Std. Deviation)')
        plt.ylabel('Expected Return')
        plt.title('Efficient Frontier')
        plt.grid(True)
        logger.debug("Efficient frontier plot generated successfully")
        plt.show()
        
    except Exception as e:
        logger.error(f"Error generating efficient frontier plot: {str(e)}")
        raise

def plot_individual_assets(processed_data):
    """
    绘制个股风险收益特征
    
    Args:
        processed_data (dict): 处理后的股票数据
    """
    price_data = pd.DataFrame()
    for ticker, data in processed_data.items():
        # 直接使用处理后的数据
        price_data[ticker] = data
    
    daily_returns = price_data.pct_change().dropna()
    mean_returns = daily_returns.mean() * 252
    volatilities = daily_returns.std() * np.sqrt(252)
    
    plt.figure(figsize=(10, 7))
    plt.scatter(volatilities, mean_returns, marker='o', s=100)
    
    for i, ticker in enumerate(mean_returns.index):
        plt.annotate(ticker, 
                    (volatilities[i], mean_returns[i]), 
                    textcoords="offset points", 
                    xytext=(0,10), 
                    ha='center')
    
    plt.xlabel('波动率 (标准差)')
    plt.ylabel('预期收益率')
    plt.title('个股风险收益分析')
    plt.grid(True)
    plt.savefig('individual_assets.png')
    plt.close()

def plot_monte_carlo_results(simulation_results, initial_investment=10000):
    """
    Creates a histogram of Monte Carlo simulation results.

    Args:
        simulation_results (numpy.ndarray): Array of simulation ending values.
        initial_investment (float, optional): Initial portfolio value.
            Defaults to 10000.

    Notes:
        Displays a histogram with KDE and marks the 5th percentile value.
    """
    ending_values = simulation_results * initial_investment
    plt.figure(figsize=(10, 6))
    sns.histplot(ending_values, bins=50, kde=True)
    plt.title('Monte Carlo Simulation Results')
    plt.xlabel('Portfolio Ending Value')
    plt.ylabel('Frequency')
    plt.axvline(x=np.percentile(ending_values, 5), color='r', linestyle='--', label='5th Percentile')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_return_distribution(portfolio_returns):
    """
    Plots the distribution of portfolio returns.

    Args:
        portfolio_returns (pandas.Series): Series of portfolio returns.

    Notes:
        Creates a histogram with kernel density estimation (KDE) of returns.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(portfolio_returns, bins=50, kde=True)
    plt.title('Distribution of Portfolio Returns')
    plt.xlabel('Daily Return')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.show()
