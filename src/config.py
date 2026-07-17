from pathlib import Path

# =============================================================================
# Project Paths
# =============================================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# =============================================================================
# Dataset Configuration
# =============================================================================

START_DATE = "2010-01-01"
END_DATE = "2024-12-31"

TRAIN_START = "2010-01-01"
TRAIN_END = "2022-12-31"

TEST_START = "2023-01-01"
TEST_END = "2024-12-31"

# =============================================================================
# Portfolio Configuration
# =============================================================================

INITIAL_CAPITAL = 1_000_000
TRANSACTION_COST = 0.0005

# =============================================================================
# Dow Jones 30 Tickers
# =============================================================================

DJIA_TICKERS = [
    "AAPL",
    "AMGN",
    "AMZN",
    "AXP",
    "BA",
    "CAT",
    "CRM",
    "CSCO",
    "CVX",
    "DIS",
    "GS",
    "HD",
    "HON",
    "IBM",
    "JNJ",
    "JPM",
    "KO",
    "MCD",
    "MMM",
    "MRK",
    "MSFT",
    "NKE",
    "PG",
    "SHW",
    "TRV",
    "UNH",
    "V",
    "VZ",
    "WMT",
    "XOM"
]