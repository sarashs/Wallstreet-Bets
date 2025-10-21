# Wallstreet-Bets Test Suite

This directory contains unit tests for the `wallstreet_quant` package.

## Running Tests

### Prerequisites

Install the test dependencies:

```bash
pip install -r requirements-dev.txt
```

You'll also need the main project dependencies (numpy, pandas, etc.):

```bash
pip install numpy pandas matplotlib seaborn yfinance networkx faiss-cpu sentence-transformers
```

### Run All Tests

```bash
pytest tests/
```

### Run Specific Test Files

```bash
pytest tests/test_utils.py
pytest tests/test_montecarlo.py
```

### Run with Coverage Report

```bash
pytest tests/ --cov=wallstreet_quant --cov-report=term-missing
```

### Run in Verbose Mode

```bash
pytest tests/ -v
```

## Test Structure

- `test_utils.py` - Tests for the `CompanyDeduper` class in `wallstreet_quant/utils.py`
- `test_montecarlo.py` - Tests for `AbstractMonteCarlo` and `NaiveMonteCarlo` classes in `wallstreet_quant/montecarlo.py`

## Current Coverage

- `wallstreet_quant/utils.py`: ~93% coverage
- `wallstreet_quant/montecarlo.py`: ~56% coverage

The tests use mocking to avoid downloading external data during test execution, making them fast and reliable.
