"""
Simple backtesting project using historical price data from Alpha Vantage.

This project explores basic momentum and moving average strategies,
with a focus on understanding data retrieval, signal generation,
backtesting methodology, and risk-adjusted performance metrics.

This is an educational project, not a trading system.
"""

import requests  # For sending HTTP requests to the Alpha Vantage API (allows the program to fetch live stock data)
import pandas as pd  # For handling tabular data (DataFrames) and performing calculations efficiently
import numpy as np  # For numerical computations (mean, std, sqrt, etc.)
import matplotlib.pyplot as plt  # For plotting stock prices, signals, and cumulative returns

# Your enterable Alpha Vantage API key (used to authenticate API requests)
API_KEY = input("Enter your Alpha Vantage API key: ").strip()

# Main loop to allow repeated analysis for multiple ticker symbols
while True:
    # Prompt user to enter a stock ticker symbol and convert to uppercase
    SYMBOL = input("Enter ticker symbol (e.g., NVDA, AAPL): ").upper()  

    # API URL to fetch daily time series data for the given symbol
    url = (
        "https://www.alphavantage.co/query"
        f"?function=TIME_SERIES_DAILY"
        f"&symbol={SYMBOL}"
        f"&outputsize=compact"
        f"&apikey={API_KEY}"
    )

    # Send GET request to the API and parse the JSON response
    response = requests.get(url)  # requests.get() sends the HTTP request
    data = response.json()        # .json() converts the response to a Python dictionary

    # Check if the API returned valid daily data
    if "Time Series (Daily)" not in data:
        print(f"\nInvalid or unavailable ticker: {SYMBOL}")
        print("Please try another symbol.\n")
        continue  # Skip to next iteration if ticker is invalid

    # Extract daily time series data
    time_series = data["Time Series (Daily)"]

    # Convert JSON data to a pandas DataFrame
    df = pd.DataFrame.from_dict(time_series, orient="index")

    # Convert index to datetime format for proper chronological sorting
    df.index = pd.to_datetime(df.index)

    # Convert all columns from strings to floats for numeric computations
    df = df.astype(float)

    # Sort the DataFrame in chronological order (oldest to newest)
    df = df.sort_index()

    # Guard against empty DataFrames
    if df.empty:
        print(f"\nNo data available for ticker: {SYMBOL}")
        print("Please try another symbol.\n")
        continue

    # Column containing closing prices (used for strategy calculations)
    CLOSE_COL = "4. close"  # *Unadjusted prices were used for simplicity; adjusted data would be more realistic*

    # MOMENTUM STRATEGY 
    df["return"] = df[CLOSE_COL].pct_change()  # Daily percentage change in price
    df["log_return"] = np.log(df[CLOSE_COL] / df[CLOSE_COL].shift(1))  # Daily log returns

    df["momentum_5"] = df[CLOSE_COL].pct_change(5)  
    # Approximates the slope of the price over 5 days, similar to a derivative in calculus

    df["signal"] = (df["momentum_5"] > 0).astype(int)  
    # Buy signal: 1 if 5-day momentum is positive, else 0

    df["strategy_return"] = df["signal"].shift(1) * df["log_return"]  
    # Apply strategy returns; shift(1) prevents look-ahead bias

    strategy = df["strategy_return"].dropna()

    cumulative_return = np.exp(strategy.cumsum()).iloc[-1] - 1  
    # Compounds daily log returns, using .iloc[-1] to avoid FutureWarning

    mean_daily = strategy.mean()  # Mean daily return
    vol_daily = strategy.std()    # Daily volatility (standard deviation)
    sharpe = mean_daily / vol_daily * np.sqrt(252) if vol_daily != 0 else np.nan  
    # Sharpe ratio measures risk-adjusted return; sqrt(252) annualizes daily volatility

    print("Cumulative return for " f"{SYMBOL}:", f"{cumulative_return:.2%}")
    print("Sharpe ratio for " f"{SYMBOL}:", f"{sharpe:.2f}")

    #  MOVING AVERAGE CROSSOVER STRATEGY 
    SHORT_WINDOW = 5
    LONG_WINDOW = 20

    df["ma_short"] = df[CLOSE_COL].rolling(SHORT_WINDOW).mean()  
    df["ma_long"] = df[CLOSE_COL].rolling(LONG_WINDOW).mean()  
    # Moving averages smooth data over a window, like a discrete approximation of an integral

    df["signal_ma"] = (df["ma_short"] > df["ma_long"]).astype(int)  
    # Buy signal: 1 if short MA > long MA, else 0

    df["strategy_ma"] = df["signal_ma"].shift(1) * df["log_return"]  
    # Apply strategy returns with shifted signal to avoid look-ahead bias

    strategy_ma = df["strategy_ma"].dropna()

    cumulative_return_ma = np.exp(strategy_ma.cumsum()).iloc[-1] - 1
    mean_daily_ma = strategy_ma.mean()
    vol_daily_ma = strategy_ma.std()
    sharpe_ma = mean_daily_ma / vol_daily_ma * np.sqrt(252) if vol_daily_ma != 0 else np.nan

    print("MA Crossover - Cumulative return for " f"{SYMBOL}:", f"{cumulative_return_ma:.2%}")
    print("MA Crossover - Sharpe ratio for " f"{SYMBOL}:", f"{sharpe_ma:.2f}")

    # PLOTTING
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df[CLOSE_COL], label="Close Price")
    plt.plot(df.index, df["ma_short"], label=f"{SHORT_WINDOW}-day MA")
    plt.plot(df.index, df["ma_long"], label=f"{LONG_WINDOW}-day MA")

    # Highlight buy signals from MA crossover
    buy_signals = df[df["signal_ma"] == 1]
    plt.scatter(buy_signals.index, buy_signals[CLOSE_COL], marker="^", color="g", label="Buy Signal")

    plt.title(f"{SYMBOL} Price & Moving Average Crossover")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.legend()
    plt.show()

    # Plot cumulative returns for both strategies
    plt.figure(figsize=(12,6))
    plt.plot(np.exp(df["strategy_return"].cumsum()), label="Momentum Strategy")
    plt.plot(np.exp(df["strategy_ma"].cumsum()), label="MA Crossover Strategy")
    plt.title(f"{SYMBOL} Cumulative Returns")
    plt.xlabel("Date")
    plt.ylabel("Cumulative Return")
    plt.legend()
    plt.show()

    # MOMENTUM PLOT 
    plt.figure(figsize=(12,6))
    plt.plot(df.index, df["momentum_5"], label="5-day Momentum", color="purple")
    plt.axhline(0, color="black", linestyle="--", linewidth=1)  # Zero line for reference
    plt.title(f"{SYMBOL} 5-Day Momentum")
    plt.xlabel("Date")
    plt.ylabel("Momentum (5-day % change)")
    plt.legend()
    plt.show()

    # LOOP CONTROL 
    again = input("\nAnalyze another symbol? (y/n): ").lower()
    if again != "y":
        print("Exiting program.")
        break

