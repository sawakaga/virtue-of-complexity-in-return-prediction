# Return Prediction Research

Python implementation to reproduce "The Virtue of Complexity in Return Prediction" by Kelly et al. (The Journal of Finance, 2023).

## Quick Start

This project uses modern Python tooling with [uv](https://docs.astral.sh/uv/) for fast dependency management.

### Prerequisites

- Python 3.11 or higher
- [uv](https://docs.astral.sh/uv/)

### Installation

```bash
# Install uv if you haven't already
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e ".[dev]"
```
### Development Setup

```bash
# Install pre-commit hooks
pre-commit install

# Run tests
pytest

# Run tests with coverage
pytest --cov=src --cov-report=html

# Format and lint code
ruff check .           # Check for issues
ruff check --fix .     # Auto-fix issues
ruff format .          # Format code
```

## Project Structure

```
.
├── src/                    # Main source code
│   └── __init__.py
├── tests/                  # Unit tests
│   └── __init__.py
├── notebooks/              # Jupyter notebooks (to be created)
├── data/                   # Data files (gitignored)
├── pyproject.toml          # Project configuration & dependencies
├── .pre-commit-config.yaml # Pre-commit hooks
└── README.md
```