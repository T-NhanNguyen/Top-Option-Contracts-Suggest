# Option Filter Helper Functions

This document describes the new helper functions added to `main.py` for filtering option contracts by date range with volume and open interest filters.

## Quick Start - Command Line Usage

The easiest way to use the filter is via command line:

```bash
# Basic usage - filter UAMY calls from 11/25 to 1/26
python main.py UAMY --date-range 11/25-1/26

# Include Greeks (delta, gamma)
python main.py UAMY --date-range 11/25-1/26 --greeks

# Filter both calls and puts
python main.py UAMY --date-range 12/1-12/31 --option-type both

# Custom volume/OI filters
python main.py UAMY --date-range 11/25-1/26 --min-volume 50 --min-oi 100

# Filter only puts with Greeks
python main.py SPY --date-range 12/15-1/15 --option-type puts --greeks
```

## Command-Line Arguments

| Argument | Description | Default |
|----------|-------------|---------|
| `ticker` | Stock ticker symbol (required) | - |
| `--date-range` | Date range in format MM/DD-MM/DD | - |
| `--option-type` | Filter calls, puts, or both | calls |
| `--greeks` | Include delta and gamma calculations | False |
| `--min-volume` | Minimum volume filter | 100 |
| `--min-oi` | Minimum open interest filter | 500 |

## Overview

Three new helper functions have been added to `main.py`:

1. **`parse_date_range(date_range_str)`** - Parses simple date format
2. **`get_filtered_options_by_date(...)`** - Retrieves and filters options by date, OI, and volume
3. **`format_options_for_chat(...)`** - Formats results for easy pasting into chat

## Function Details

### 1. parse_date_range(date_range_str)

Parses a simple date range format into datetime objects.

**Format**: `MM/DD-MM/DD` (e.g., `"11/25-1/26"`)

**Parameters**:
- `date_range_str` (str): Date range string

**Returns**:
- Tuple of (start_date, end_date) as datetime objects

**Behavior**:
- Assumes current year for start date
- Handles year rollover (e.g., `11/25-1/26` spans from November to January)
- If start date is in the past, assumes next year

**Example**:
```python
start, end = parse_date_range("11/25-1/26")
# Returns: (2025-11-25, 2026-01-26)
```

---

### 2. get_filtered_options_by_date(ticker, date_range, ...)

Retrieves option contracts within a date range, filtered by volume and open interest.

**Parameters**:
- `ticker` (str): Stock ticker symbol (e.g., "UAMY")
- `date_range` (str): Date range in format "MM/DD-MM/DD"
- `min_volume` (int): Minimum volume filter (default: 100)
- `min_oi` (int): Minimum open interest filter (default: 500)
- `option_type` (str): "calls", "puts", or "both" (default: "calls")
- `include_greeks` (bool): Whether to calculate Greeks (delta, gamma) (default: False)

**Returns**:
Dictionary containing:
- `ticker`: Stock ticker
- `current_price`: Current stock price
- `date_range`: Original date range string
- `start_date`: Parsed start date (YYYY-MM-DD)
- `end_date`: Parsed end date (YYYY-MM-DD)
- `filters`: Applied filters (min_volume, min_oi, option_type)
- `options_count`: Number of contracts found
- `options`: List of option contracts with details

**Option Contract Fields**:
- `strike`: Strike price
- `type`: "call" or "put"
- `expiration`: Expiration date (YYYY-MM-DD)
- `dte`: Days to expiration
- `last_price`: Last traded price
- `iv`: Implied volatility
- `oi`: Open interest
- `volume`: Trading volume
- `bid`: Bid price
- `ask`: Ask price
- `gamma`: Gamma (if include_greeks=True)
- `delta`: Delta (if include_greeks=True)

**Example**:
```python
results = get_filtered_options_by_date(
    ticker="UAMY",
    date_range="11/25-1/26",
    min_volume=100,
    min_oi=500,
    option_type="calls",
    include_greeks=True
)
```

---

### 3. format_options_for_chat(results, include_greeks=False)

Formats the results from `get_filtered_options_by_date()` for easy pasting into chat applications.

**Parameters**:
- `results` (dict): Results from `get_filtered_options_by_date()`
- `include_greeks` (bool): Whether to include Greeks in formatted output (default: False)

**Returns**:
- String formatted for chat/messaging applications

**Output Format**:
```
📊 UAMY Option Contracts
Current Price: $1.23
Date Range: 2025-11-25 to 2026-01-26
Filters: Volume≥100, OI≥500
Found 15 contracts

📅 2025-12-20 (DTE: 45)
------------------------------------------------------------
Type  Strike   Price    IV      OI        Vol
CALL  $1.50    $0.15    45.2%   1,234     567
CALL  $2.00    $0.08    52.1%   2,345     890

📅 2026-01-17 (DTE: 73)
------------------------------------------------------------
Type  Strike   Price    IV      OI        Vol
CALL  $1.50    $0.25    48.5%   1,890     456
CALL  $2.00    $0.12    55.3%   3,456     1,234
```

**Example**:
```python
formatted = format_options_for_chat(results, include_greeks=True)
print(formatted)
```

---

## Complete Usage Example

```python
from main import get_filtered_options_by_date, format_options_for_chat, clear_cache

# Clear cache for fresh data
clear_cache()

# Get UAMY calls expiring between 11/25 and 1/26
results = get_filtered_options_by_date(
    ticker="UAMY",
    date_range="11/25-1/26",
    min_volume=100,
    min_oi=500,
    option_type="calls",
    include_greeks=True
)

# Format for chat and print
if "error" not in results:
    formatted = format_options_for_chat(results, include_greeks=True)
    print(formatted)
else:
    print(f"Error: {results['error']}")
```

---

## Use Cases

### 1. Quick Option Screening for Chat Discussion
```python
results = get_filtered_options_by_date("UAMY", "12/1-12/31", min_volume=50, min_oi=100)
print(format_options_for_chat(results))
# Copy output and paste into "Options strategy for UAMY calls" chat
```

### 2. Find High-Liquidity Options in Specific Timeframe
```python
results = get_filtered_options_by_date(
    ticker="SPY",
    date_range="11/15-11/30",
    min_volume=1000,
    min_oi=5000,
    option_type="both"
)
```

### 3. Analyze Greeks for Near-Term Expiries
```python
results = get_filtered_options_by_date(
    ticker="AAPL",
    date_range="11/20-12/20",
    min_volume=500,
    min_oi=1000,
    option_type="calls",
    include_greeks=True
)
formatted = format_options_for_chat(results, include_greeks=True)
```

---

## Notes

- The functions leverage existing codebase functions for OI/volume filtering
- Uses `get_option_chain_analysis_optimized()` for data retrieval
- Greeks (delta, gamma) are calculated using existing `calculate_delta()` and `calculate_gamma()` functions
- Output is sorted by expiration date, then by strike price
- Date parsing handles year rollovers automatically
- All filters are applied at the option contract level

---

## Error Handling

Functions return error dictionaries when issues occur:

```python
results = get_filtered_options_by_date("INVALID", "11/25-1/26")
# Returns: {"error": "Failed to get stock info for INVALID"}
```

Common errors:
- Invalid ticker symbol
- Invalid date range format
- No options found matching criteria
- Network/API errors from yfinance

---

## Getting Help

View all available command-line options:

```bash
python main.py --help
```

This will show:
- All available arguments for date filter mode
- Full analysis mode options
- Usage examples
