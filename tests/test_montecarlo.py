import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch, MagicMock
from wallstreet_quant.montecarlo import AbstractMonteCarlo, NaiveMonteCarlo


class TestAbstractMonteCarlo:
    """Test suite for AbstractMonteCarlo base class."""
    
    def test_init(self):
        """Test initialization."""
        # Create a concrete implementation for testing
        class ConcreteMonteCarlo(AbstractMonteCarlo):
            def run_simulation(self, days, n_simulations):
                return {}
        
        tickers = ["AAPL", "MSFT"]
        mc = ConcreteMonteCarlo(tickers)
        assert mc.tickers == tickers
        assert mc.data is None
        assert mc.log_returns is None
    
    @patch('wallstreet_quant.montecarlo.yf.download')
    def test_fetch_data_single_ticker(self, mock_download):
        """Test fetching data for a single ticker."""
        class ConcreteMonteCarlo(AbstractMonteCarlo):
            def run_simulation(self, days, n_simulations):
                return {}
        
        # Mock data with MultiIndex columns
        dates = pd.date_range('2024-01-01', periods=5)
        data = pd.DataFrame({
            ('Close', 'AAPL'): [100, 101, 102, 103, 104]
        }, index=dates)
        data.columns = pd.MultiIndex.from_tuples([('Close', 'AAPL')])
        mock_download.return_value = data
        
        mc = ConcreteMonteCarlo(["AAPL"])
        result = mc.fetch_data(period="1y")
        
        assert result is not None
        assert mc.data is not None
    
    @patch('wallstreet_quant.montecarlo.yf.download')
    def test_fetch_data_multiple_tickers(self, mock_download):
        """Test fetching data for multiple tickers."""
        class ConcreteMonteCarlo(AbstractMonteCarlo):
            def run_simulation(self, days, n_simulations):
                return {}
        
        # Mock data with MultiIndex columns
        dates = pd.date_range('2024-01-01', periods=5)
        data = pd.DataFrame({
            ('Close', 'AAPL'): [100, 101, 102, 103, 104],
            ('Close', 'MSFT'): [200, 201, 202, 203, 204]
        }, index=dates)
        data.columns = pd.MultiIndex.from_tuples([('Close', 'AAPL'), ('Close', 'MSFT')])
        mock_download.return_value = data
        
        mc = ConcreteMonteCarlo(["AAPL", "MSFT"])
        result = mc.fetch_data(period="1y")
        
        assert result is not None
        assert mc.data is not None
        assert mc.data.shape[1] == 2
    
    def test_compute_log_returns(self):
        """Test computation of log returns."""
        class ConcreteMonteCarlo(AbstractMonteCarlo):
            def run_simulation(self, days, n_simulations):
                return {}
        
        mc = ConcreteMonteCarlo(["AAPL"])
        mc.data = pd.DataFrame({
            'AAPL': [100, 110, 105, 115]
        })
        
        log_returns = mc.compute_log_returns()
        
        assert log_returns is not None
        assert len(log_returns) == 3  # One less than data points
        assert isinstance(log_returns, pd.DataFrame)
    
    def test_extract_means_and_covariances(self):
        """Test extraction of means and covariances."""
        class ConcreteMonteCarlo(AbstractMonteCarlo):
            def run_simulation(self, days, n_simulations):
                return {}
        
        mc = ConcreteMonteCarlo(["AAPL", "MSFT"])
        mc.data = pd.DataFrame({
            'AAPL': [100, 101, 102, 103, 104],
            'MSFT': [200, 201, 202, 203, 204]
        })
        mc.compute_log_returns()
        
        means, covariances = mc.extract_means_and_covariances()
        
        assert means is not None
        assert covariances is not None
        assert len(means) == 2
        assert covariances.shape == (2, 2)
    
    def test_get_summary_stats(self):
        """Test getting summary statistics."""
        class ConcreteMonteCarlo(AbstractMonteCarlo):
            def run_simulation(self, days, n_simulations):
                return {}
        
        mc = ConcreteMonteCarlo(["AAPL"])
        mc.data = pd.DataFrame({
            'AAPL': [100, 101, 102, 103, 104]
        })
        mc.compute_log_returns()
        mc.extract_means_and_covariances()
        
        stats = mc.get_summary_stats()
        
        assert 'mean_returns' in stats
        assert 'volatility' in stats
        assert 'correlation_matrix' in stats
        assert 'data_points' in stats


class TestNaiveMonteCarlo:
    """Test suite for NaiveMonteCarlo class."""
    
    @patch('wallstreet_quant.montecarlo.yf.download')
    def test_init_default_weights(self, mock_download):
        """Test initialization with default weights."""
        # Mock data
        dates = pd.date_range('2024-01-01', periods=5)
        data = pd.DataFrame({
            ('Close', 'AAPL'): [100, 101, 102, 103, 104],
            ('Close', 'MSFT'): [200, 201, 202, 203, 204]
        }, index=dates)
        data.columns = pd.MultiIndex.from_tuples([('Close', 'AAPL'), ('Close', 'MSFT')])
        mock_download.return_value = data
        
        tickers = ["AAPL", "MSFT"]
        mc = NaiveMonteCarlo(tickers)
        
        assert len(mc.weights) == len(tickers)
        np.testing.assert_almost_equal(np.sum(mc.weights), 1.0)
        np.testing.assert_array_almost_equal(mc.weights, [0.5, 0.5])
    
    @patch('wallstreet_quant.montecarlo.yf.download')
    def test_init_custom_weights(self, mock_download):
        """Test initialization with custom weights."""
        # Mock data
        dates = pd.date_range('2024-01-01', periods=5)
        data = pd.DataFrame({
            ('Close', 'AAPL'): [100, 101, 102, 103, 104],
            ('Close', 'MSFT'): [200, 201, 202, 203, 204]
        }, index=dates)
        data.columns = pd.MultiIndex.from_tuples([('Close', 'AAPL'), ('Close', 'MSFT')])
        mock_download.return_value = data
        
        tickers = ["AAPL", "MSFT"]
        weights = [0.7, 0.3]
        mc = NaiveMonteCarlo(tickers, weights=weights)
        
        np.testing.assert_array_almost_equal(mc.weights, [0.7, 0.3])
    
    @patch('wallstreet_quant.montecarlo.yf.download')
    def test_init_with_alpha_correction(self, mock_download):
        """Test initialization with alpha correction."""
        # Mock data
        dates = pd.date_range('2024-01-01', periods=5)
        data = pd.DataFrame({
            ('Close', 'AAPL'): [100, 101, 102, 103, 104]
        }, index=dates)
        data.columns = pd.MultiIndex.from_tuples([('Close', 'AAPL')])
        mock_download.return_value = data
        
        tickers = ["AAPL"]
        alpha_correction = [0.1]  # 10% annualized return
        mc = NaiveMonteCarlo(tickers, alpha_correction=alpha_correction)
        
        assert mc.alpha_correction == alpha_correction
        # Mean should be adjusted
        assert mc.means is not None
    
    @patch('wallstreet_quant.montecarlo.yf.download')
    def test_run_simulation(self, mock_download):
        """Test running Monte Carlo simulation."""
        # Mock data
        dates = pd.date_range('2024-01-01', periods=100)
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
        data = pd.DataFrame({
            ('Close', 'AAPL'): prices
        }, index=dates)
        data.columns = pd.MultiIndex.from_tuples([('Close', 'AAPL')])
        mock_download.return_value = data
        
        mc = NaiveMonteCarlo(["AAPL"])
        results = mc.run_simulation(days=30, n_simulations=1000)
        
        assert results is not None
        assert 'daily_returns' in results
        assert 'cumulative_log_returns' in results
        assert 'final_log_returns' in results
        assert 'final_actual_returns' in results
        assert 'portfolio_mean' in results
        assert 'portfolio_std' in results
        
        # Check shapes
        assert results['daily_returns'].shape == (1000, 30)
        assert results['cumulative_log_returns'].shape == (1000, 30)
        assert len(results['final_log_returns']) == 1000
        assert len(results['final_actual_returns']) == 1000
    
    @patch('wallstreet_quant.montecarlo.yf.download')
    def test_get_simulation_summary(self, mock_download):
        """Test getting simulation summary."""
        # Mock data
        dates = pd.date_range('2024-01-01', periods=100)
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
        data = pd.DataFrame({
            ('Close', 'AAPL'): prices
        }, index=dates)
        data.columns = pd.MultiIndex.from_tuples([('Close', 'AAPL')])
        mock_download.return_value = data
        
        mc = NaiveMonteCarlo(["AAPL"])
        mc.run_simulation(days=30, n_simulations=1000)
        summary = mc.get_simulation_summary()
        
        assert 'mean_return_pct' in summary
        assert 'std_return_pct' in summary
        assert 'percentiles' in summary
        assert 'var_5_pct' in summary
        assert 'cvar_5_pct' in summary
        assert 'probability_positive' in summary
        
        # Check percentiles
        assert '5th' in summary['percentiles']
        assert '50th' in summary['percentiles']
        assert '95th' in summary['percentiles']
    
    @patch('wallstreet_quant.montecarlo.yf.download')
    def test_simulation_reproducibility(self, mock_download):
        """Test that simulation is reproducible with the same seed."""
        # Mock data
        dates = pd.date_range('2024-01-01', periods=100)
        np.random.seed(42)
        prices = 100 * np.exp(np.cumsum(np.random.normal(0.001, 0.02, 100)))
        data = pd.DataFrame({
            ('Close', 'AAPL'): prices
        }, index=dates)
        data.columns = pd.MultiIndex.from_tuples([('Close', 'AAPL')])
        mock_download.return_value = data
        
        mc1 = NaiveMonteCarlo(["AAPL"])
        results1 = mc1.run_simulation(days=30, n_simulations=100)
        
        mc2 = NaiveMonteCarlo(["AAPL"])
        results2 = mc2.run_simulation(days=30, n_simulations=100)
        
        # Results should be identical due to fixed seed
        np.testing.assert_array_almost_equal(
            results1['final_log_returns'], 
            results2['final_log_returns']
        )
