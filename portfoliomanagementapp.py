import streamlit as st
import pandas as pd
import yfinance as yf
import riskfolio as rp
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
from scipy.stats import norm
import warnings
warnings.filterwarnings("ignore")
import os


# Set up the app
st.title("Portfolio Management and Analysis App")

# Function to load existing portfolios
def load_portfolios():
    if os.path.exists("portfolios.csv"):
        return pd.read_csv("portfolios.csv")
    else:
        return pd.DataFrame(columns=["Portfolio Name", "Tickers"])

# Function to save portfolios
def save_portfolio(portfolio_name, tickers):
    portfolios = load_portfolios()
    new_portfolio = pd.DataFrame({"Portfolio Name": [portfolio_name], "Tickers": [tickers]})
    portfolios = pd.concat([portfolios, new_portfolio], ignore_index=True)
    portfolios.to_csv("portfolios.csv", index=False)

# Function to update portfolios
def update_portfolio(portfolio_name, new_tickers):
    portfolios = load_portfolios()
    portfolios.loc[portfolios['Portfolio Name'] == portfolio_name, 'Tickers'] = new_tickers
    portfolios.to_csv("portfolios.csv", index=False)

# Function to fetch stock data
def fetch_stock_data(tickers):
    tickers_list = [ticker.strip() for ticker in tickers.split(',')]
    data = yf.download(tickers_list)['Adj Close'].dropna()
    return data

# Function to delete a portfolio
def delete_portfolio(portfolio_name):
    portfolios = load_portfolios()
    portfolios = portfolios[portfolios["Portfolio Name"] != portfolio_name]
    portfolios.to_csv("portfolios.csv", index=False)
    st.success(f"Portfolio '{portfolio_name}' deleted!")

# Function to make covariance matrix positive definite
def make_positive_definite(cov_matrix):
    """ Adjusts the covariance matrix to ensure it is positive definite """
    # Compute the eigenvalues and eigenvectors
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    
    # Adjust eigenvalues: set any eigenvalues less than a small threshold to a small positive number
    min_eigenvalue = 1e-10
    adjusted_eigenvalues = np.maximum(eigenvalues, min_eigenvalue)
    
    # Reconstruct the covariance matrix
    adjusted_cov_matrix = np.dot(eigenvectors, np.dot(np.diag(adjusted_eigenvalues), eigenvectors.T))
    
    return adjusted_cov_matrix

# Function to calculate portfolio standard deviation based on the number of assets
def calculate_portfolio_std(num_assets, avg_correlation, asset_volatility):
    return asset_volatility * np.sqrt((1 / num_assets) + avg_correlation * (1 - (1 / num_assets)))

# Function to calculate rolling 12-month CAGR
def calculate_rolling_cagr(returns, window):
    rolling_cagr = (1 + returns).rolling(window=window).apply(lambda x: np.prod(x) ** (252/window) - 1)
    return rolling_cagr

# Function to calculate rolling Sharpe ratio
def calculate_rolling_sharpe_ratio(returns, risk_free_rate, window):
    excess_returns = returns - (risk_free_rate / 252)
    rolling_sharpe = excess_returns.rolling(window=window).mean() / excess_returns.rolling(window=window).std() * np.sqrt(252)
    return rolling_sharpe

# Performance metric functions
def sharpe_ratio(return_series, N, rf):
    mean = return_series.mean() * N - rf
    sigma = return_series.std() * np.sqrt(N)
    return mean / sigma

def sortino_ratio(return_series, N, rf):
    mean = return_series.mean() * N - rf
    std_neg = return_series[return_series < 0].std() * np.sqrt(N)
    return mean / std_neg

def max_drawdown(return_series):
    comp_ret = (return_series + 1).cumprod()
    peak = comp_ret.expanding(min_periods=1).max()
    dd = (comp_ret / peak) - 1
    return dd.min()

def calculate_cvar(returns, confidence_level=0.95):
    sorted_returns = np.sort(returns)
    index = int((1 - confidence_level) * len(sorted_returns))
    cvar = sorted_returns[:index].mean()
    return cvar

def calculate_portfolio_metrics(returns, N=252, rf=0.01):
    metrics = {}
    total_return = (returns + 1).prod() - 1
    annualized_return = (1 + total_return) ** (N / len(returns)) - 1
    std_dev = returns.std() * np.sqrt(N)
    sharpe = sharpe_ratio(returns, N, rf)
    sortino = sortino_ratio(returns, N, rf)
    max_dd = max_drawdown(returns)
    calmar = annualized_return / abs(max_dd)
    cvar = calculate_cvar(returns)

    metrics['Total Return'] = total_return * 100
    metrics['Annualized Return'] = annualized_return * 100
    metrics['Standard Deviation'] = std_dev * 100
    metrics['Sharpe Ratio'] = sharpe
    metrics['Sortino Ratio'] = sortino
    metrics['Max Drawdown'] = max_dd * 100
    metrics['Calmar Ratio'] = calmar
    metrics['CVar'] = cvar

    return metrics


# Display saved portfolios
st.header("Select a Portfolio")
portfolios = load_portfolios()
if not portfolios.empty:
    with st.expander("Saved Portfolios"):
        # Scrollable section for portfolios
        st.markdown(
            """
            <style>
            .scrollable-portfolios {
                height: 200px;
                overflow-y: scroll;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

        st.markdown('<div class="scrollable-portfolios">', unsafe_allow_html=True)

        for index, row in portfolios.iterrows():
            col1, col2, col3 = st.columns([6, 1, 1])
            col1.write(row["Portfolio Name"])
            if col2.button("Edit", key=f"edit_{index}"):
                st.session_state.edit_portfolio = row["Portfolio Name"]
            if col3.button("Delete", key=f"delete_{index}"):
                delete_portfolio(row["Portfolio Name"])
                st.experimental_rerun()

        st.markdown('</div>', unsafe_allow_html=True)

        if st.button("+ Create New Portfolio"):
            st.session_state.create_new = True

# Create input form for new portfolio
if st.session_state.get("create_new", False):
    st.header("Create a New Portfolio")
    with st.form("portfolio_form"):
        portfolio_name = st.text_input("Portfolio Name")
        tickers = st.text_area("Tickers (comma-separated)", placeholder="e.g., AAPL, MSFT, GOOGL")
        submitted = st.form_submit_button("Save Portfolio")

        if submitted:
            if portfolio_name and tickers:
                save_portfolio(portfolio_name, tickers)
                st.success(f"Portfolio '{portfolio_name}' saved!")
                st.session_state.create_new = False
                st.experimental_rerun()  # Refresh the app to show updated portfolio list
            else:
                st.error("Please provide both portfolio name and ticker symbols.")

# Edit portfolio functionality
if "edit_portfolio" in st.session_state:
    st.header(f"Edit Portfolio: {st.session_state.edit_portfolio}")
    with st.form("edit_form"):
        edit_name = st.session_state.edit_portfolio
        current_tickers = portfolios[portfolios["Portfolio Name"] == edit_name]["Tickers"].values[0]
        new_tickers = st.text_area("New Tickers (comma-separated)", value=current_tickers)
        edit_submitted = st.form_submit_button("Update Portfolio")

        if edit_submitted:
            if new_tickers:
                update_portfolio(edit_name, new_tickers)
                st.success(f"Portfolio '{edit_name}' updated!")
                del st.session_state.edit_portfolio
                st.experimental_rerun()  # Refresh the app to show updated portfolio list
            else:
                st.error("Please provide new ticker symbols.")

# Display selected portfolio data
if not portfolios.empty:
    selected_portfolio = st.selectbox("Select a portfolio to fetch data", portfolios["Portfolio Name"].unique())
    if selected_portfolio:
        tickers = portfolios[portfolios["Portfolio Name"] == selected_portfolio]["Tickers"].values[0]
        stock_data = fetch_stock_data(tickers)
        st.write(f"## Stock Data for {selected_portfolio}")
        st.dataframe(stock_data)

        # Plot the data
        st.write("### Price Trends")
        plt.figure(figsize=(10, 5))
        for column in stock_data.columns:
            plt.plot(stock_data.index, stock_data[column], label=column)
        plt.legend()
        plt.title(f"Stock Price Trends for {selected_portfolio}")
        plt.xlabel("Date")
        plt.ylabel("Adjusted Close Price")
        st.pyplot(plt)

        # Calculate returns
        returns = stock_data.pct_change().dropna()

        # Display correlation matrix
        st.write("### Correlation Matrix")
        corr_matrix = returns.corr()
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', ax=ax)
        st.pyplot(fig)

        # Calculate annualized standard deviation (volatility)
        st.write("### Annualized Standard Deviation (Volatility)")
        individual_asset_stds = []

        # Calculate individual asset standard deviations
        for ticker in returns.columns:
            annualized_std = returns[ticker].std() * np.sqrt(252)
            individual_asset_stds.append(annualized_std)
        
        average_asset_std = np.mean(individual_asset_stds)
        avg_corr = np.mean([corr_matrix.iloc[i, j] for i in range(len(corr_matrix)) for j in range(i+1, len(corr_matrix))])

        # Select optimization model
        st.write("## Portfolio Optimization")
        optimization_model = st.selectbox("Choose Optimization Model", ["Hierarchical Risk Parity (HRP)", "Classic Risk Parity (CRP)", "Efficient Frontier (EF)"])

        if optimization_model:
            if optimization_model == "Hierarchical Risk Parity (HRP)":
                port = rp.HCPortfolio(returns=returns)
                weights = port.optimization(
                    model="HRP",
                    obj='Sharpe',
                    codependence="pearson",
                    rm="MV",
                    rf=0,
                    linkage="single",
                    max_k=10,
                    leaf_order=True,
                )

                # Plot the dendrogram
                st.write("### Dendrogram")
                fig, ax = plt.subplots()
                rp.plot_dendrogram(
                    returns=returns,
                    codependence="pearson",
                    linkage="single",
                    k=None,
                    max_k=10,
                    leaf_order=True,
                    ax=ax,
                )
                st.pyplot(fig)

                # Plot the portfolio weights
                st.write("### Portfolio Weights")
                fig, ax = plt.subplots()
                rp.plot_pie(
                    w=weights,
                    title="HRP Risk Parity",
                    others=0.05,
                    nrow=25,
                    cmap="tab20",
                    height=6,
                    width=10,
                    ax=ax,
                )
                st.pyplot(fig)

                # Plot the risk contributions
                st.write("### Risk Contributions")
                fig, ax = plt.subplots()
                rp.plot_risk_con(
                    w=weights,
                    cov=returns.cov(),
                    returns=returns,
                    rm="MV",
                    rf=0,
                    alpha=0.05,
                    color="tab:blue",
                    height=6,
                    width=10,
                    t_factor=252,
                    ax=ax,
                )
                st.pyplot(fig)

                # Plot the histogram of portfolio returns
                st.write("### Histogram of Portfolio Returns")
                fig, ax = plt.subplots()
                rp.plot_hist(
                    returns=returns,
                    w=weights,
                    alpha=0.05,
                    bins=50,
                    height=6,
                    width=10,
                    ax=ax
                )
                st.pyplot(fig)

                # Calculate portfolio returns
                portfolio_returns = returns.dot(weights)

                # Aligning start date with portfolio returns
                start = portfolio_returns.index[0]  # First date of the portfolio returns
                end = datetime.now()

                # Define the benchmarks
                benchmarks = ['SPY', 'QQQ']

                # Download the benchmark data
                benchmark_data = yf.download(benchmarks, start=start, end=end)['Adj Close']

                # Calculate daily returns
                benchmark_returns = benchmark_data.pct_change().dropna()

                # Reindex benchmark returns to match portfolio returns
                benchmark_returns = benchmark_returns.reindex(portfolio_returns.index).dropna()

                # Add portfolio returns as a new column to the benchmark returns
                all_returns = benchmark_returns.copy()
                all_returns['HRP Portfolio'] = portfolio_returns

                # Calculate performance metrics
                metrics = ['Total Return', 'Annualized Return', 'Standard Deviation', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'CVar', 'Maximum Drawdown', 'Kurtosis', 'Skewness']
                performance_df = pd.DataFrame(index=metrics, columns=benchmarks + ['HRP Portfolio'])

                N = 252

                # Total Return (Cumulative Return)
                performance_df.loc['Total Return'] = (all_returns + 1).prod() - 1

                # Annualized Return
                performance_df.loc['Annualized Return'] = (1 + performance_df.loc['Total Return']) ** (N / len(all_returns)) - 1

                # Standard Deviation
                performance_df.loc['Standard Deviation'] = all_returns.std() * np.sqrt(N)

                # Sharpe Ratio
                performance_df.loc['Sharpe Ratio'] = all_returns.apply(sharpe_ratio, args=(N, rf,))

                # Sortino Ratio
                performance_df.loc['Sortino Ratio'] = all_returns.apply(sortino_ratio, args=(N, rf,))

                # Calmar Ratio
                max_drawdowns = all_returns.apply(max_drawdown)
                performance_df.loc['Calmar Ratio'] = all_returns.mean() * N / abs(max_drawdowns)

                # CVaR
                performance_df.loc['CVar'] = all_returns.apply(calculate_cvar)

                # Maximum Drawdown
                performance_df.loc['Maximum Drawdown'] = max_drawdowns

                # Kurtosis
                performance_df.loc['Kurtosis'] = all_returns.kurtosis()

                # Skewness
                performance_df.loc['Skewness'] = all_returns.skew()

                performance_df.loc['Performance Starting Date'] = start.strftime('%d/%m/%Y')

                # Format as percentages with 2 decimal places for specific metrics
                percentage_metrics = ['Total Return', 'Annualized Return', 'Standard Deviation', 'CVar', 'Maximum Drawdown']
                performance_df.loc[percentage_metrics] = performance_df.loc[percentage_metrics].applymap(lambda x: f"{x:.2%}")

                # Format other metrics as floats with 2 decimal places
                float_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Kurtosis', 'Skewness']
                performance_df.loc[float_metrics] = performance_df.loc[float_metrics].applymap(lambda x: f"{x:.2f}")

                st.table(performance_df)

            elif optimization_model == "Classic Risk Parity (CRP)":
                # Perform Classic Risk Parity optimization
                port = rp.Portfolio(returns=returns)
                port.assets_stats(method_mu='hist', method_cov='hist')
                port.cov = make_positive_definite(port.cov)
                weights = port.rp_optimization(
                    model="Classic",
                    rm="MV",
                    hist=True,
                    rf=0,
                    b=None
                )

                # Plot the portfolio weights
                st.write("### Portfolio Weights")
                fig, ax = plt.subplots()
                rp.plot_pie(
                    w=weights,
                    title="Classic Risk Parity",
                    others=0.05,
                    nrow=25,
                    cmap="tab20",
                    height=8,
                    width=10,
                    ax=ax,
                )
                st.pyplot(fig)

                # Plot the risk contributions
                st.write("### Risk Contributions")
                fig, ax = plt.subplots()
                rp.plot_risk_con(
                    w=weights,
                    cov=port.cov,
                    returns=port.returns,
                    rm="MV",
                    rf=0,
                    alpha=0.05,
                    color="tab:blue",
                    height=6,
                    width=10,
                    t_factor=252,
                    ax=ax,
                )
                st.pyplot(fig)

                # Plot the histogram of portfolio returns
                st.write("### Histogram of Portfolio Returns")
                fig, ax = plt.subplots()
                rp.plot_hist(
                    returns=returns,
                    w=weights,
                    alpha=0.05,
                    bins=50,
                    height=6,
                    width=10,
                    ax=ax
                )
                st.pyplot(fig)

                # Calculate portfolio returns
                portfolio_returns = returns.dot(weights)

                # Aligning start date with portfolio returns
                start = portfolio_returns.index[0]  # First date of the portfolio returns
                end = datetime.now()

                # Define the benchmarks
                benchmarks = ['SPY', 'QQQ']

                # Download the benchmark data
                benchmark_data = yf.download(benchmarks, start=start, end=end)['Adj Close']

                # Calculate daily returns
                benchmark_returns = benchmark_data.pct_change().dropna()

                # Reindex benchmark returns to match portfolio returns
                benchmark_returns = benchmark_returns.reindex(portfolio_returns.index).dropna()

                # Add portfolio returns as a new column to the benchmark returns
                all_returns = benchmark_returns.copy()
                all_returns['CRP Portfolio'] = portfolio_returns

                # Calculate performance metrics
                metrics = ['Total Return', 'Annualized Return', 'Standard Deviation', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'CVar', 'Maximum Drawdown', 'Kurtosis', 'Skewness']
                performance_df = pd.DataFrame(index=metrics, columns=benchmarks + ['CRP Portfolio'])

                rf_data = pdr.get_data_fred('DGS1MO', start=start, end=end).interpolate()
                rf = rf_data.iloc[-1, 0] / 100  # Last available 1-Month Treasury rate as risk-free rate

                N = 252

                # Total Return (Cumulative Return)
                performance_df.loc['Total Return'] = (all_returns + 1).prod() - 1

                # Annualized Return
                performance_df.loc['Annualized Return'] = (1 + performance_df.loc['Total Return']) ** (N / len(all_returns)) - 1

                # Standard Deviation
                performance_df.loc['Standard Deviation'] = all_returns.std() * np.sqrt(N)

                # Sharpe Ratio
                performance_df.loc['Sharpe Ratio'] = all_returns.apply(sharpe_ratio, args=(N, rf,))

                # Sortino Ratio
                performance_df.loc['Sortino Ratio'] = all_returns.apply(sortino_ratio, args=(N, rf,))

                # Calmar Ratio
                max_drawdowns = all_returns.apply(max_drawdown)
                performance_df.loc['Calmar Ratio'] = all_returns.mean() * N / abs(max_drawdowns)

                # CVaR
                performance_df.loc['CVar'] = all_returns.apply(calculate_cvar)

                # Maximum Drawdown
                performance_df.loc['Maximum Drawdown'] = max_drawdowns

                # Kurtosis
                performance_df.loc['Kurtosis'] = all_returns.kurtosis()

                # Skewness
                performance_df.loc['Skewness'] = all_returns.skew()

                performance_df.loc['Performance Starting Date'] = start.strftime('%d/%m/%Y')

                # Format as percentages with 2 decimal places for specific metrics
                percentage_metrics = ['Total Return', 'Annualized Return', 'Standard Deviation', 'CVar', 'Maximum Drawdown']
                performance_df.loc[percentage_metrics] = performance_df.loc[percentage_metrics].applymap(lambda x: f"{x:.2%}")

                # Format other metrics as floats with 2 decimal places
                float_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Kurtosis', 'Skewness']
                performance_df.loc[float_metrics] = performance_df.loc[float_metrics].applymap(lambda x: f"{x:.2f}")

                st.table(performance_df)

            elif optimization_model == "Efficient Frontier (EF)":
                # Perform Efficient Frontier optimization
                method_mu = 'hist'
                method_cov = 'hist'
                hist = True
                model = 'Classic'
                rm = 'MV'
                obj = 'Sharpe'
                rf = 0
                l = 0

                port = rp.Portfolio(returns=returns)
                port.assets_stats(method_mu=method_mu, method_cov=method_cov)
                weights = port.optimization(
                    model=model,
                    rm=rm,
                    obj=obj,
                    rf=rf,
                    l=l,
                    hist=hist
                )

                # Plot the portfolio weights
                st.write("### Portfolio Weights")
                fig, ax = plt.subplots()
                rp.plot_pie(
                    w=weights,
                    title="Optimum Portfolio",
                    others=0.05,
                    cmap="tab20",
                    ax=ax,
                )
                st.pyplot(fig)

                # Plot the efficient frontier
                st.write("### Efficient Frontier")
                frontier = port.efficient_frontier(
                    model=model,
                    rm=rm,
                    points=50,
                    rf=rf,
                    hist=hist
                )

                fig, ax = plt.subplots()
                rp.plot_frontier(
                    w_frontier=frontier,
                    mu=port.mu,
                    cov=port.cov,
                    returns=returns,
                    rm=rm,
                    rf=rf,
                    cmap='viridis',
                    w=weights,
                    ax=ax
                )
                st.pyplot(fig)

                fig, ax = plt.subplots()
                rp.plot_frontier_area(
                    w_frontier=frontier,
                    cmap="tab20",
                    ax=ax
                )
                st.pyplot(fig)

                # Plot the histogram of portfolio returns
                st.write("### Histogram of Portfolio Returns")
                fig, ax = plt.subplots()
                rp.plot_hist(
                    returns=returns,
                    w=weights,
                    alpha=0.05,
                    bins=50,
                    height=6,
                    width=10,
                    ax=ax
                )
                st.pyplot(fig)

                # Calculate portfolio returns
                portfolio_returns = returns.dot(weights)

                # Aligning start date with portfolio returns
                start = portfolio_returns.index[0]  # First date of the portfolio returns
                end = datetime.now()

                # Define the benchmarks
                benchmarks = ['SPY', 'QQQ']

                # Download the benchmark data
                benchmark_data = yf.download(benchmarks, start=start, end=end)['Adj Close']

                # Calculate daily returns
                benchmark_returns = benchmark_data.pct_change().dropna()

                # Reindex benchmark returns to match portfolio returns
                benchmark_returns = benchmark_returns.reindex(portfolio_returns.index).dropna()

                # Add portfolio returns as a new column to the benchmark returns
                all_returns = benchmark_returns.copy()
                all_returns['EF Portfolio'] = portfolio_returns

                # Calculate performance metrics
                metrics = ['Total Return', 'Annualized Return', 'Standard Deviation', 'Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'CVar', 'Maximum Drawdown', 'Kurtosis', 'Skewness']
                performance_df = pd.DataFrame(index=metrics, columns=benchmarks + ['EF Portfolio'])

                rf_data = pdr.get_data_fred('DGS1MO', start=start, end=end).interpolate()
                rf = rf_data.iloc[-1, 0] / 100  # Last available 1-Month Treasury rate as risk-free rate

                N = 252

                # Total Return (Cumulative Return)
                performance_df.loc['Total Return'] = (all_returns + 1).prod() - 1

                # Annualized Return
                performance_df.loc['Annualized Return'] = (1 + performance_df.loc['Total Return']) ** (N / len(all_returns)) - 1

                # Standard Deviation
                performance_df.loc['Standard Deviation'] = all_returns.std() * np.sqrt(N)

                # Sharpe Ratio
                performance_df.loc['Sharpe Ratio'] = all_returns.apply(sharpe_ratio, args=(N, rf,))

                # Sortino Ratio
                performance_df.loc['Sortino Ratio'] = all_returns.apply(sortino_ratio, args=(N, rf,))

                # Calmar Ratio
                max_drawdowns = all_returns.apply(max_drawdown)
                performance_df.loc['Calmar Ratio'] = all_returns.mean() * N / abs(max_drawdowns)

                # CVaR
                performance_df.loc['CVar'] = all_returns.apply(calculate_cvar)

                # Maximum Drawdown
                performance_df.loc['Maximum Drawdown'] = max_drawdowns

                # Kurtosis
                performance_df.loc['Kurtosis'] = all_returns.kurtosis()

                # Skewness
                performance_df.loc['Skewness'] = all_returns.skew()

                performance_df.loc['Performance Starting Date'] = start.strftime('%d/%m/%Y')

                # Format as percentages with 2 decimal places for specific metrics
                percentage_metrics = ['Total Return', 'Annualized Return', 'Standard Deviation', 'CVar', 'Maximum Drawdown']
                performance_df.loc[percentage_metrics] = performance_df.loc[percentage_metrics].applymap(lambda x: f"{x:.2%}")

                # Format other metrics as floats with 2 decimal places
                float_metrics = ['Sharpe Ratio', 'Sortino Ratio', 'Calmar Ratio', 'Kurtosis', 'Skewness']
                performance_df.loc[float_metrics] = performance_df.loc[float_metrics].applymap(lambda x: f"{x:.2f}")

                st.table(performance_df)

            # Calculate portfolio standard deviation (annualized) using optimized weights
            portfolio_std = np.sqrt(np.dot(weights.T, np.dot(returns.cov() * 252, weights)))

            st.write(f"Portfolio Standard Deviation (Annualized): {portfolio_std.item():.2%}")
            st.write(f"Average Single Asset Standard Deviation: {average_asset_std:.2%}")

            # Create a DataFrame to hold the additional information
            weight_df = pd.DataFrame(weights, columns=["weights"])
            weight_df["weights"] = weight_df["weights"] * 100
            weight_df["weights"] = weight_df["weights"].map('{:.2f}%'.format)

            tickers_info = []
            for ticker in weight_df.index:
                ticker_info = yf.Ticker(ticker).info
                tickers_info.append({
                    "Ticker": ticker,
                    "Name": ticker_info.get("longName", "N/A"),
                    "Sector": ticker_info.get("sector", "N/A"),
                    "Industry": ticker_info.get("industry", "N/A"),
                })
            
            st.write("### Portfolio Composition")

            tickers_info_df = pd.DataFrame(tickers_info)
            weight_df = weight_df.merge(tickers_info_df, left_index=True, right_on="Ticker")
            st.table(weight_df)

            # Plot the historical compounded cumulative returns of the portfolio
            st.write("### Historical Compounded Cumulative Returns")
            fig, ax = plt.subplots(figsize=(10, 8))
            rp.plot_drawdown(
                returns=returns,
                w=weights,
                alpha=0.05,
                ax=ax
            )
            st.pyplot(fig)

            # Calculate rolling 12-month CAGR and Sharpe ratio
            rolling_cagr = calculate_rolling_cagr(returns.dot(weights), window=252)
            rolling_sharpe = calculate_rolling_sharpe_ratio(returns.dot(weights), risk_free_rate=0.01, window=252)

            st.write("### Rolling 12-Month CAGR and Sharpe Ratio")
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(rolling_cagr, label="Rolling 12-Month CAGR")
            ax.set_ylabel("CAGR", color='tab:blue')
            ax.tick_params(axis='y', labelcolor='tab:blue')
            ax2 = ax.twinx()
            ax2.plot(rolling_sharpe, label="Rolling Sharpe Ratio", color='tab:red')
            ax2.set_ylabel("Sharpe Ratio", color='tab:red')
            ax2.tick_params(axis='y', labelcolor='tab:red')
            ax.grid()
            fig.tight_layout()
            st.pyplot(fig)
            
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# Section for computing and plotting the Holy Grail of Investing
st.header("Holy Grail of Investing")

# User inputs for average correlation and asset volatility
avg_correlation = st.number_input("Average Correlation", value=0.02, min_value=0.0, max_value=1.0, step=0.01)
asset_volatility = st.number_input("Asset Volatility (Annualized)", value=0.45, min_value=0.0, max_value=1.0, step=0.01)

# Calculate portfolio standard deviation for different numbers of assets
num_assets_range = range(1, 26)
portfolio_stds_user = [calculate_portfolio_std(n, avg_correlation, asset_volatility) for n in num_assets_range]

# Calculate portfolio standard deviation for the selected portfolio's average correlation and volatility
portfolio_stds_portfolio = [calculate_portfolio_std(n, avg_corr, average_asset_std) for n in num_assets_range]

# Calculate the average annual return of the assets in the portfolio
average_annual_return = returns.mean().mean() * 252

# Calculate the Return to Risk Ratio
def calculate_return_to_risk_ratio(return_rate, std_dev):
    return return_rate / std_dev

return_to_risk_ratios = [calculate_return_to_risk_ratio(average_annual_return, std) for std in portfolio_stds_user]

# Calculate the probability of losing money in a given year
def calculate_probability_of_losing_money(return_to_risk_ratio):
    return norm.cdf(-return_to_risk_ratio)

probability_of_losing_money = [calculate_probability_of_losing_money(rtr) for rtr in return_to_risk_ratios]

# Plot the portfolio standard deviations
fig, ax1 = plt.subplots()

# Primary y-axis: Portfolio Standard Deviation
ax1.plot(num_assets_range, np.array(portfolio_stds_user) * 100, label=f"User: Correlation = {avg_correlation}, Asset Volatility = {asset_volatility}")
ax1.plot(num_assets_range, np.array(portfolio_stds_portfolio) * 100, label=f"Portfolio: Correlation = {avg_corr:.2f}, Asset Volatility = {average_asset_std:.2%}", linestyle='--')
ax1.set_xlabel("Number of Assets in Portfolio")
ax1.set_ylabel("Portfolio Standard Deviation (%)")
ax1.set_title("Portfolio Standard Deviation (assets equal in weight, variance and covariance)")
ax1.legend(loc="upper right")
ax1.grid(True)

# Secondary y-axis: Return to Risk Ratio (transparent line)
ax2 = ax1.twinx()
ax2.plot(num_assets_range, return_to_risk_ratios, color='tab:green', alpha=0)  # Make the line fully transparent
ax2.set_ylabel("Return to Risk Ratio", color='tab:green')
ax2.tick_params(axis='y', labelcolor='tab:green')
ax2.invert_yaxis()  # Invert the y-axis for Return to Risk Ratio

# Third y-axis: Probability of Losing Money in a Given Year (transparent line)
ax3 = ax1.twinx()
rspine = ax3.spines['right']
rspine.set_position(('axes', 1.15))
ax3.plot(num_assets_range, np.array(probability_of_losing_money) * 100, color='tab:red', alpha=0)  # Make the line fully transparent
ax3.set_ylabel("Probability of Losing Money (%)", color='tab:red')
ax3.tick_params(axis='y', labelcolor='tab:red')
ax3.grid(False)

fig.tight_layout()
st.pyplot(fig)


