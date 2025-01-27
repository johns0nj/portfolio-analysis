# main.py #push to  https://github.com/johns0nj/portfolio-analysis test
from data_fetching import fetch_stock_data
from data_processing import process_stock_data
from calculations import calculate_portfolio_metrics, monte_carlo_simulation, calculate_portfolio_returns
from visualization import (
    plot_portfolio_performance,
    plot_efficient_frontier,
    plot_individual_assets,
    plot_monte_carlo_results,
    plot_return_distribution
)
from logger_config import setup_logger
import numpy as np

"""
Main module for portfolio analysis application.

This module orchestrates the portfolio analysis process by combining
data fetching, processing, calculations, and visualization components.
It serves as the entry point for the application.
"""

def main():
    """
    Main function to run the portfolio analysis.

    The function performs the following steps:
    1. Fetches historical stock data
    2. Processes and cleans the data
    3. Calculates portfolio metrics
    4. Generates visualizations
    5. Runs Monte Carlo simulations

    Raises:
        Exception: If any step in the analysis process fails.
    """
    # Setup logger
    logger = setup_logger()
    logger.info("Starting portfolio analysis")

    try:
        # User inputs
        portfolio = {'AAPL': 0.4, 'MSFT': 0.3, 'GOOGL': 0.3}
        start_date = '2020-01-01'
        end_date = '2024-12-31'
        
        logger.info(f"Analyzing portfolio: {portfolio}")
        logger.info(f"Time period: {start_date} to {end_date}")

        # Fetch data
        logger.debug("Fetching stock data...")
        raw_data = fetch_stock_data(list(portfolio.keys()), start_date, end_date)
        
        # 获取基准数据
        logger.debug("Fetching benchmark data...")
        try:
            benchmark_data = fetch_stock_data(['^GSPC'], start_date, end_date)
            benchmark_returns = benchmark_data['^GSPC'].pct_change().dropna()
        except Exception as e:
            logger.warning(f"无法获取基准数据: {str(e)}")
            benchmark_returns = None

        # Process data
        logger.debug("Processing stock data...")
        processed_data = process_stock_data(raw_data)

        # Calculate metrics
        logger.debug("Calculating portfolio metrics...")
        metrics = calculate_portfolio_metrics(processed_data, portfolio)
        portfolio_returns = calculate_portfolio_returns(processed_data, portfolio)
        
        # 计算相对基准的表现
        if benchmark_returns is not None:
            try:
                excess_return = (portfolio_returns.mean() - benchmark_returns.mean()) * 252
                tracking_error = (portfolio_returns - benchmark_returns).std() * np.sqrt(252)
                information_ratio = excess_return / float(tracking_error) if tracking_error > 0 else None
            except Exception as e:
                logger.warning(f"计算相对基准表现时出错: {str(e)}")
                excess_return = tracking_error = information_ratio = None
        else:
            excess_return = tracking_error = information_ratio = None
        
        # 添加新的指标
        metrics['Excess Return'] = excess_return
        metrics['Tracking Error'] = tracking_error
        metrics['Information Ratio'] = information_ratio
        
        logger.info("Portfolio Metrics:")
        for key, value in metrics.items():
            if value is None:
                logger.info(f"{key}: Not available")
            else:
                logger.info(f"{key}: {value:.4f}")

        # Visualizations
        logger.debug("Generating visualizations...")
        plot_portfolio_performance(portfolio_returns, benchmark_returns)  # 添加基准对比
        plot_individual_assets(processed_data)
        plot_efficient_frontier(processed_data)
        plot_return_distribution(portfolio_returns)

        # Monte Carlo Simulation
        logger.debug("Running Monte Carlo simulation...")
        simulation_results = monte_carlo_simulation(processed_data, portfolio)
        plot_monte_carlo_results(simulation_results)
        
        logger.info("Portfolio analysis completed successfully")

    except Exception as e:
        logger.error(f"An error occurred during portfolio analysis: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()
