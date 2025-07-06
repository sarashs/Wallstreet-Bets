import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from abc import ABC, abstractmethod
from typing import List, Dict, Tuple
import seaborn as sns

class AbstractMonteCarlo(ABC):
    """
    Abstract base class for Monte Carlo simulations.
    Handles data fetching and basic statistics computation.
    """
    
    def __init__(self, tickers: List[str]):
        """
        Initialize with a list of stock tickers.
        
        Args:
            tickers: List of stock ticker symbols
        """
        self.tickers = tickers
        self.data = None
        self.log_returns = None
        self.means = None
        self.covariances = None
        
    def fetch_data(self, period: str = "1y") -> pd.DataFrame:
        """
        Fetch stock data from Yahoo Finance.
        
        Args:
            period: Time period for data (default: 1y)
            
        Returns:
            DataFrame with adjusted close prices
        """
        try:
            data = yf.download(self.tickers, period=period, progress=False)
            if data is None or data.empty:
                print("No data retrieved")
                return None
                
            if len(self.tickers) == 1:
                # For single ticker, yfinance returns a DataFrame with MultiIndex columns
                if isinstance(data.columns, pd.MultiIndex):
                    if 'Close' in data.columns.levels[0]:
                        self.data = data['Close'].copy()
                        self.data.columns = self.tickers
                    else:
                        print("'Close' column not found in data")
                        return None
                else:
                    # Fallback for older yfinance versions
                    if 'Close' in data.columns:
                        self.data = data[['Close']].copy()
                        self.data.columns = self.tickers
                    else:
                        print("'Close' column not found in data")
                        return None
            else:
                # For multiple tickers, check if it's a MultiIndex DataFrame
                if isinstance(data.columns, pd.MultiIndex):
                    if 'Close' in data.columns.levels[0]:
                        self.data = data['Close'].copy()
                    else:
                        print("'Close' not found in MultiIndex columns")
                        return None
                else:
                    # Fallback: assume it's already the right format
                    self.data = data.copy()
            
            return self.data
        except Exception as e:
            print(f"Error fetching data: {e}")
            return None
    
    def compute_log_returns(self) -> pd.DataFrame:
        """
        Compute log returns from price data.
        
        Returns:
            DataFrame with log returns
        """
        if self.data is None:
            print("No data available. Fetching data first...")
            self.fetch_data()
        
        if self.data is None:
            print("Failed to fetch data. Cannot compute log returns.")
            return None
        
        self.log_returns = np.log(self.data / self.data.shift(1)).dropna()
        return self.log_returns
    
    def extract_means_and_covariances(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Extract means and covariances from log returns.
        
        Returns:
            Tuple of (means, covariance matrix)
        """
        if self.log_returns is None:
            self.compute_log_returns()
        
        if self.log_returns is None:
            print("No log returns available. Cannot compute means and covariances.")
            return None, None
        
        self.means = self.log_returns.mean().values
        self.covariances = self.log_returns.cov().values
        
        return self.means, self.covariances
    
    def get_summary_stats(self) -> Dict:
        """
        Get summary statistics for the data.
        
        Returns:
            Dictionary with summary statistics
        """
        if self.log_returns is None:
            self.compute_log_returns()
        
        return {
            'mean_returns': self.means,
            'volatility': np.sqrt(np.diag(self.covariances)),
            'correlation_matrix': self.log_returns.corr().values,
            'data_points': len(self.log_returns)
        }
    
    @abstractmethod
    def run_simulation(self, days: int, n_simulations: int) -> Dict:
        """
        Abstract method for running Monte Carlo simulation.
        
        Args:
            days: Number of days to simulate
            n_simulations: Number of simulation runs
            
        Returns:
            Dictionary with simulation results
        """
        pass


class NaiveMonteCarlo(AbstractMonteCarlo):
    """
    Naive Monte Carlo simulation implementation.
    Inherits from AbstractMonteCarlo and implements Monte Carlo simulation.
    """
    
    def __init__(self, tickers: List[str], alpha_correction: List[float] = None, period="1y"):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            tickers: List of stock ticker symbols
            alpha_correction: Optional list of annualized yearly return. Alpha corrections for each ticker to correct for estimated future portfolio alpha (default: None)
        """
        super().__init__(tickers)
        if alpha_correction is not None:
            assert len(tickers) == len(alpha_correction), "Length of tickers and alpha_correction must match."
        self.alpha_correction = alpha_correction
        self.simulation_results = None
        self.fetch_data(period)
        # Ensure we have means and covariances
        if self.means is None or self.covariances is None:
            self.extract_means_and_covariances()
        if self.alpha_correction is not None:
            # Adjust means with alpha correction if provided
            self.means += np.log(np.array(self.alpha_correction)+1)/252  # Convert alpha correction: annualized return to daily log returns
    
    def run_simulation(self, days: int, n_simulations: int = 1000) -> Dict:
        """
        Run Monte Carlo simulation for log returns.
        
        Args:
            days: Number of days to simulate
            n_simulations: Number of simulation runs (default: 1000)
            
        Returns:
            Dictionary with simulation results
        """

        # Generate random samples
        np.random.seed(42)  # For reproducibility
        
        # For portfolio simulation, we'll use equal weights
        n_assets = len(self.tickers)
        weights = np.ones(n_assets) / n_assets
        
        # Calculate portfolio mean and variance
        portfolio_mean = np.dot(weights, self.means)
        portfolio_variance = np.dot(weights, np.dot(self.covariances, weights))
        portfolio_std = np.sqrt(portfolio_variance)
        
        # Generate random returns for each simulation
        daily_returns = np.random.normal(portfolio_mean, portfolio_std, (n_simulations, days))
        
        # Calculate cumulative log returns
        cumulative_log_returns = np.cumsum(daily_returns, axis=1)
        
        # Final log returns (at the end of simulation period)
        final_log_returns = cumulative_log_returns[:, -1]
        
        # Convert to actual returns (exponential of log returns minus 1)
        final_actual_returns = np.exp(final_log_returns) - 1
        
        # Store results
        self.simulation_results = {
            'daily_returns': daily_returns,
            'cumulative_log_returns': cumulative_log_returns,
            'final_log_returns': final_log_returns,
            'final_actual_returns': final_actual_returns,
            'portfolio_mean': portfolio_mean,
            'portfolio_std': portfolio_std,
            'days': days,
            'n_simulations': n_simulations
        }
        
        return self.simulation_results
    
    def plot_results(self, show_percentiles: bool = True):
        """
        Plot Monte Carlo simulation results.
        
        Args:
            show_percentiles: Whether to show percentile lines
        """
        if self.simulation_results is None:
            print("No simulation results to plot. Run simulation first.")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # Plot 1: Sample paths of cumulative log returns
        sample_paths = self.simulation_results['cumulative_log_returns'][:1000]  # Show first 1000 paths
        days = self.simulation_results['days']
        
        axes[0, 0].plot(range(1, days + 1), sample_paths.T, alpha=0.1, color='blue')
        axes[0, 0].set_title('Sample Paths of Cumulative Log Returns')
        axes[0, 0].set_xlabel('Days')
        axes[0, 0].set_ylabel('Cumulative Log Return')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Distribution of final log returns
        final_log_returns = self.simulation_results['final_log_returns']
        axes[0, 1].hist(final_log_returns, bins=50, alpha=0.7, color='green', edgecolor='black')
        axes[0, 1].set_title('Distribution of Final Log Returns')
        axes[0, 1].set_xlabel('Final Log Return')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Distribution of final actual returns (in percent)
        final_actual_returns_pct = self.simulation_results['final_actual_returns'] * 100
        axes[1, 0].hist(final_actual_returns_pct, bins=50, alpha=0.7, color='red', edgecolor='black')
        axes[1, 0].set_title('Distribution of Final Actual Returns (%)')
        axes[1, 0].set_xlabel('Final Actual Return (%)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Add percentile lines if requested
        if show_percentiles:
            percentiles = [5, 25, 50, 75, 95]
            colors = ['red', 'orange', 'green', 'orange', 'red']
            
            for p, color in zip(percentiles, colors):
                value = np.percentile(final_actual_returns_pct, p)
                axes[1, 0].axvline(value, color=color, linestyle='--', alpha=0.8, 
                                 label=f'{p}th percentile: {value:.2f}%')
        
        axes[1, 0].legend()
        
        # Plot 4: Risk metrics summary
        axes[1, 1].axis('off')
        
        # Calculate key statistics
        stats_text = f"""
        Simulation Summary:
        
        Portfolio: {', '.join(self.tickers)}
        Simulation Days: {days}
        Number of Simulations: {self.simulation_results['n_simulations']:,}
        
        Final Returns Statistics:
        Mean Return: {np.mean(final_actual_returns_pct):.2f}%
        Std Deviation: {np.std(final_actual_returns_pct):.2f}%
        
        Percentiles:
        5th:  {np.percentile(final_actual_returns_pct, 5):.2f}%
        25th: {np.percentile(final_actual_returns_pct, 25):.2f}%
        50th: {np.percentile(final_actual_returns_pct, 50):.2f}%
        75th: {np.percentile(final_actual_returns_pct, 75):.2f}%
        95th: {np.percentile(final_actual_returns_pct, 95):.2f}%
        
        Risk Metrics:
        VaR (5%): {np.percentile(final_actual_returns_pct, 5):.2f}%
        CVaR (5%): {np.mean(final_actual_returns_pct[final_actual_returns_pct <= np.percentile(final_actual_returns_pct, 5)]):.2f}%
        """
        
        axes[1, 1].text(0.1, 0.9, stats_text, transform=axes[1, 1].transAxes, 
                        fontsize=10, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.show()
    
    def get_simulation_summary(self) -> Dict:
        """
        Get a summary of simulation results.
        
        Returns:
            Dictionary with key statistics
        """
        if self.simulation_results is None:
            print("No simulation results. Run simulation first.")
            return {}
        
        final_actual_returns_pct = self.simulation_results['final_actual_returns'] * 100
        
        return {
            'mean_return_pct': np.mean(final_actual_returns_pct),
            'std_return_pct': np.std(final_actual_returns_pct),
            'percentiles': {
                '5th': np.percentile(final_actual_returns_pct, 5),
                '25th': np.percentile(final_actual_returns_pct, 25),
                '50th': np.percentile(final_actual_returns_pct, 50),
                '75th': np.percentile(final_actual_returns_pct, 75),
                '95th': np.percentile(final_actual_returns_pct, 95)
            },
            'var_5_pct': np.percentile(final_actual_returns_pct, 5),
            'cvar_5_pct': np.mean(final_actual_returns_pct[final_actual_returns_pct <= np.percentile(final_actual_returns_pct, 5)]),
            'probability_positive': np.mean(final_actual_returns_pct > 0) * 100
        }


if __name__ == "__main__":
    # Example with a few tech stocks
    tickers = ['JBL', 'HPQ', 'MRK']

    # Create Monte Carlo simulator
    mc = NaiveMonteCarlo(tickers)

    # Fetch data and compute statistics
    print("Fetching data...")
    mc.fetch_data(period="2y")

    print("Computing log returns and statistics...")
    mc.extract_means_and_covariances()

    # Display summary statistics
    stats = mc.get_summary_stats()
    print("\nSummary Statistics:")
    print(f"Mean daily log returns: {stats['mean_returns']}")
    print(f"Daily volatilities: {stats['volatility']}")
    print(f"Data points: {stats['data_points']}")

    # Run Monte Carlo simulation
    print("\nRunning Monte Carlo simulation...")
    results = mc.run_simulation(days=252, n_simulations=100000)  # 1 year simulation

    # Display results summary
    summary = mc.get_simulation_summary()
    print("\nSimulation Results Summary:")
    print(f"Mean return: {summary['mean_return_pct']:.2f}%")
    print(f"Standard deviation: {summary['std_return_pct']:.2f}%")
    print(f"5th percentile (VaR): {summary['var_5_pct']:.2f}%")
    print(f"Probability of positive return: {summary['probability_positive']:.1f}%")

    # Plot results
    mc.plot_results(show_percentiles=True)