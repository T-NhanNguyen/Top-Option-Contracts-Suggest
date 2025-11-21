import yfinance as yf
import pandas as pd
from datetime import datetime
import time
from functools import lru_cache
import concurrent.futures
from typing import Dict, List, Optional
import threading

# Global cache with thread safety
_cache_lock = threading.RLock()
_stock_info_cache: Dict[str, Dict] = {}
_option_chain_cache: Dict[str, Dict] = {}
_expiration_cache: Dict[str, List] = {}

# Rate limiting configuration
MAX_RETRIES = 3
RETRY_DELAY = 1  # seconds

MAX_BID_ASK_SPREAD = 0.3  # Maximum acceptable bid/ask spread as a percentage of last price

@lru_cache(maxsize=100)
def get_cached_stock_info(ticker: str) -> Optional[Dict]:
    """Cache stock info with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            with _cache_lock:
                if ticker in _stock_info_cache:
                    return _stock_info_cache[ticker]
                
                stock = yf.Ticker(ticker)
                info = stock.info
                _stock_info_cache[ticker] = info
                return info
                
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"Failed to get stock info for {ticker}: {str(e)}")
                return None
            time.sleep(RETRY_DELAY * (attempt + 1))

@lru_cache(maxsize=100)
def get_cached_expirations(ticker: str) -> List[str]:
    """Cache expiration dates with retry logic"""
    for attempt in range(MAX_RETRIES):
        try:
            with _cache_lock:
                if ticker in _expiration_cache:
                    return _expiration_cache[ticker]
                
                stock = yf.Ticker(ticker)
                expirations = stock.options
                _expiration_cache[ticker] = expirations
                return expirations
                
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"Failed to get expirations for {ticker}: {str(e)}")
                return []
            time.sleep(RETRY_DELAY * (attempt + 1))

def get_cached_option_chain(ticker: str, expiration: str) -> Optional[Dict]:
    """Cache option chain data with retry logic"""
    cache_key = f"{ticker}_{expiration}"
    
    for attempt in range(MAX_RETRIES):
        try:
            with _cache_lock:
                if cache_key in _option_chain_cache:
                    return _option_chain_cache[cache_key]
                
                stock = yf.Ticker(ticker)
                chain = stock.option_chain(expiration)
                
                # Store only essential data to save memory
                cached_data = {
                    'calls': chain.calls[['contractSymbol', 'strike', 'openInterest', 'impliedVolatility', 'lastPrice', 'volume', 'bid', 'ask']],
                    'puts': chain.puts[['contractSymbol', 'strike', 'openInterest', 'impliedVolatility', 'lastPrice', 'volume', 'bid', 'ask']]
                }
                
                _option_chain_cache[cache_key] = cached_data
                return cached_data
                
        except Exception as e:
            if attempt == MAX_RETRIES - 1:
                print(f"Failed to get option chain for {ticker} {expiration}: {str(e)}")
                return None
            time.sleep(RETRY_DELAY * (attempt + 1))

def process_expiration_batch(ticker: str, expirations: List[str], min_dte: int, today: datetime) -> List[Dict]:
    """Process multiple expirations in parallel"""
    results = []
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        # Create future tasks
        future_to_expiry = {
            executor.submit(process_single_expiration, ticker, expiry, min_dte, today): expiry 
            for expiry in expirations
        }
        
        # Collect results as they complete
        for future in concurrent.futures.as_completed(future_to_expiry):
            result = future.result()
            if result:
                results.append(result)
    
    return results

def process_single_expiration(ticker: str, expiration: str, min_dte: int, today: datetime) -> Optional[Dict]:
    """Process a single expiration with lazy evaluation"""
    # Precompute DTE once
    expiry_date = datetime.strptime(expiration, '%Y-%m-%d')
    dte = (expiry_date - today).days  # Use datetime objects, not dates
    
    if dte < min_dte:
        return None
    
    # Get cached option chain data
    chain_data = get_cached_option_chain(ticker, expiration)
    if not chain_data:
        return None
    
    calls = chain_data['calls'].copy()
    puts = chain_data['puts'].copy()
    for df in [calls, puts]:
        if df.empty:
            continue
        # Ensure required columns exist with defaults
        required_cols = ['contractSymbol', 'strike', 'openInterest', 'impliedVolatility', 'lastPrice', 'volume', 'bid', 'ask']
        for col in required_cols:
            if col not in df.columns:
                if col == 'contractSymbol':
                    df[col] = ''
                else:
                    df[col] = 0.0 if col in ['impliedVolatility', 'lastPrice', 'bid', 'ask'] else 0
        # Filter by bid/ask spread
        df['bid_ask_spread'] = (df['ask'] - df['bid']) / df['lastPrice']
        df = df[df['bid_ask_spread'] <= MAX_BID_ASK_SPREAD]
    # Now filter safely
    if calls.empty and puts.empty:
        return None
    top_calls = calls.nlargest(5, 'openInterest') if not calls.empty else pd.DataFrame()
    top_puts = puts.nlargest(5, 'openInterest') if not puts.empty else pd.DataFrame()
    
    # Bulk IV processing
    calls_iv = top_calls['impliedVolatility'].mean() if not top_calls.empty else 0
    puts_iv = top_puts['impliedVolatility'].mean() if not top_puts.empty else 0
    avg_iv = (calls_iv + puts_iv) / 2 if (calls_iv and puts_iv) else max(calls_iv, puts_iv)
    
    return {
        'expiration': expiration,
        'dte': dte,
        'calls': top_calls.to_dict('records'),
        'puts': top_puts.to_dict('records'),
        'total_oi': top_calls['openInterest'].sum() + top_puts['openInterest'].sum(),
        'total_volume': top_calls['volume'].sum() + top_puts['volume'].sum(),
        'avg_iv': avg_iv
    }

def get_put_call_ratio(ticker: str, expiration: str = None):
    """
    Calculate put/call ratio using existing cached option chain data.
    
    Returns:
    - Volume P/C Ratio: Total put volume / Total call volume
    - OI P/C Ratio: Total put open interest / Total call open interest
    """
    if expiration:
        # Single expiration
        chain = get_cached_option_chain(ticker, expiration)
        if not chain:
            return {"error": f"No data for {ticker} {expiration}"}
        
        calls = chain['calls']
        puts = chain['puts']
        
        call_volume = calls['volume'].sum()
        put_volume = puts['volume'].sum()
        call_oi = calls['openInterest'].sum()
        put_oi = puts['openInterest'].sum()
        
        return {
            'expiration': expiration,
            'volume_pcr': put_volume / call_volume if call_volume > 0 else 0,
            'oi_pcr': put_oi / call_oi if call_oi > 0 else 0,
            'call_volume': int(call_volume),
            'put_volume': int(put_volume),
            'call_oi': int(call_oi),
            'put_oi': int(put_oi)
        }
    else:
        # All expirations
        expirations = get_cached_expirations(ticker)
        total_call_vol = 0
        total_put_vol = 0
        total_call_oi = 0
        total_put_oi = 0
        
        for exp in expirations:
            chain = get_cached_option_chain(ticker, exp)
            if chain:
                total_call_vol += chain['calls']['volume'].sum()
                total_put_vol += chain['puts']['volume'].sum()
                total_call_oi += chain['calls']['openInterest'].sum()
                total_put_oi += chain['puts']['openInterest'].sum()
        
        return {
            'ticker': ticker,
            'volume_pcr': total_put_vol / total_call_vol if total_call_vol > 0 else 0,
            'oi_pcr': total_put_oi / total_call_oi if total_call_oi > 0 else 0,
            'total_call_volume': int(total_call_vol),
            'total_put_volume': int(total_put_vol),
            'total_call_oi': int(total_call_oi),
            'total_put_oi': int(total_put_oi)
        }

def get_option_chain_analysis_optimized(ticker: str, min_dte: int = 45) -> Dict:
    """
    Optimized option chain analysis with all requested enhancements
    """
    try:
        # Precompute time values once
        today = datetime.now()
        
        # Get cached data with retry logic
        stock_info = get_cached_stock_info(ticker)
        if not stock_info:
            return {"error": f"Failed to get stock info for {ticker}"}
        
        current_price = stock_info.get('regularMarketPrice', stock_info.get('currentPrice', 0))
        
        # Get cached expirations
        all_expirations = get_cached_expirations(ticker)
        if not all_expirations:
            return {"error": f"No options data available for {ticker}"}
        
        # Process expirations in parallel batches
        expiration_results = process_expiration_batch(ticker, all_expirations, min_dte, today)
        
        if not expiration_results:
            return {"error": f"No expirations found with DTE >= {min_dte} for {ticker}"}
        
        # Lazy aggregation of results
        total_oi = sum(result['total_oi'] for result in expiration_results)
        total_volume = sum(result['total_volume'] for result in expiration_results)
        
        # Prepare final response with lazy evaluation
        analysis = {
            "ticker": ticker,
            "current_price": current_price,
            "analysis_date": today.strftime('%Y-%m-%d'),
            "min_dte": min_dte,
            "qualified_expirations_count": len(expiration_results),
            "total_open_interest": total_oi,
            "total_volume": total_volume,
            "expiration_data": {result['expiration']: result for result in expiration_results}
        }
        
        return analysis
        
    except Exception as e:
        return {"error": f"Error analyzing {ticker}: {str(e)}"}

# Cache management utilities
def clear_cache():
    """Clear all caches"""
    with _cache_lock:
        _stock_info_cache.clear()
        _option_chain_cache.clear()
        _expiration_cache.clear()
        get_cached_stock_info.cache_clear()
        get_cached_expirations.cache_clear()

def get_cache_stats() -> Dict:
    """Get cache statistics"""
    with _cache_lock:
        return {
            "stock_info_cache_size": len(_stock_info_cache),
            "option_chain_cache_size": len(_option_chain_cache),
            "expiration_cache_size": len(_expiration_cache)
        }

# Example usage with optimized features
def test_optimized_features():
    ticker = "AAPL"
    min_dte = 45
    
    print(f"Running OPTIMIZED option chain analysis for {ticker}...")
    print("=" * 60)
    print(f"Cache stats before: {get_cache_stats()}")
    
    # First call - will make API calls and populate cache
    start_time = time.time()
    analysis = get_option_chain_analysis_optimized(ticker, min_dte)
    first_call_time = time.time() - start_time
    
    print(f"First call completed in {first_call_time:.2f} seconds")
    print(f"Cache stats after first call: {get_cache_stats()}")
    
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
    else:
        print(f"\nAnalysis for {analysis['ticker']} at ${analysis['current_price']:.2f}")
        print(f"Qualified expirations: {analysis['qualified_expirations_count']}")
        print(f"Total OI: {analysis['total_open_interest']:,}")
        print(f"Total Volume: {analysis['total_volume']:,}")
        
        # Second call - should be much faster due to caching
        print("\n" + "=" * 60)
        print("MAKING SECOND CALL (should be faster due to caching)...")
        
        start_time = time.time()
        analysis2 = get_option_chain_analysis_optimized(ticker, min_dte)
        second_call_time = time.time() - start_time
        
        print(f"Second call completed in {second_call_time:.2f} seconds")
        print(f"Speed improvement: {first_call_time/second_call_time:.1f}x faster")
        
        # Test with different parameters to show cache efficiency
        print("\n" + "=" * 60)
        print("TESTING WITH DIFFERENT DTE PARAMETERS...")
        
        for test_dte in [30, 60, 90]:
            start_time = time.time()
            test_analysis = get_option_chain_analysis_optimized(ticker, test_dte)
            test_time = time.time() - start_time
            
            if "error" not in test_analysis:
                print(f"DTE {test_dte}: {test_analysis['qualified_expirations_count']} expirations, {test_time:.3f}s")
        
        # Display sample data (lazy evaluation in action)
        print("\n" + "=" * 60)
        print("SAMPLE EXPIRATION DATA (first 2 expirations):")
        
        expiration_count = 0
        for expiry, data in list(analysis['expiration_data'].items())[:2]:
            print(f"\n{expiry} (DTE: {data['dte']}) - Total OI: {data['total_oi']:,}")
            print(f"  Avg IV: {data['avg_iv']:.2%}")
            
            if data['calls']:
                sample_call = data['calls'][0]
                print(f"  Sample Call: ${sample_call['strike']} - {sample_call['openInterest']:,} OI")
            
            expiration_count += 1
            if expiration_count >= 2:
                break
        
        # Test rate limiting by simulating multiple rapid calls
        print("\n" + "=" * 60)
        print("TESTING RATE LIMITING HANDLING...")
        
        test_tickers = ["AAPL", "MSFT", "SPY", "QQQ", "NVDA"]
        results = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            future_to_ticker = {
                executor.submit(get_option_chain_analysis_optimized, ticker, min_dte): ticker 
                for ticker in test_tickers
            }
            
            for future in concurrent.futures.as_completed(future_to_ticker):
                ticker = future_to_ticker[future]
                try:
                    result = future.result()
                    results.append((ticker, result))
                    print(f"✓ Completed: {ticker}")
                except Exception as e:
                    print(f"✗ Failed: {ticker} - {str(e)}")
        
        print(f"\nFinal cache stats: {get_cache_stats()}")
        
        # Clear cache demonstration
        print("\n" + "=" * 60)
        print("DEMONSTRATING CACHE CLEARING...")
        
        clear_cache()
        print(f"Cache after clearing: {get_cache_stats()}")

def test_putcall_ratio():
    expiration_date = "2025-11-21"
    print("\n" + "=" * 60)
    print("TESTING PUT/CALL RATIO CALCULATION...")
    
    sectors_data = {
        "Materials": ["XLB", "VAW", "IYM"],
        "Comm. Services": ["XLC", "VOX", "IYZ"],
        "Energy": ["XLE", "VDE", "IYE"],
        "Financials": ["XLF", "VFH", "IYF"],
        "Industrials": ["XLI", "VIS", "IYJ"],
        "Technology": ["XLK", "VGT", "IYW"],
        "Consumer Staples": ["XLP", "VDC", "IYK"],
        "Consumer Discret.": ["XLY", "VCR", "IYC"],
        "Health Care": ["XLV", "VHT", "IYH"],
        "Utilities": ["XLU", "VPU", "IDU"],
        "Real Estate": ["XLRE", "VNQ", "IYR"]
    }
    providers = ["SPDR", "Vanguard", "iShares", "Average"]
    all_tickers = []
    for sector_tickers in sectors_data.values():
        all_tickers.extend(sector_tickers)
    
    results = {}  # Will store {ticker: ratio_dict} for successful fetches
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        future_to_ticker = {
            executor.submit(get_put_call_ratio, ticker, expiration_date): ticker 
            for ticker in all_tickers
        }
        
        for future in concurrent.futures.as_completed(future_to_ticker):
            ticker = future_to_ticker[future]
            try:
                result = future.result()
                results[ticker] = result
                # Check if result is an error
                if 'error' in result:
                    print(f"✗ Failed: {ticker} - {result['error']}")
                else:
                    print(f"✓ Completed: {ticker}")
            except Exception as e:
                print(f"✗ Failed: {ticker} - {str(e)}")
                # Store None values for failed tickers
                results[ticker] = {'volume_pcr': None, 'oi_pcr': None}
    
    # Build table data: dict of sectors -> list of [vol_pcr, oi_pcr] per provider + average
    table_data = {}
    for sector, tickers in sectors_data.items():
        vols = []
        ois = []
        sector_results = []
        for i, ticker in enumerate(tickers):
            ratio = results.get(ticker, {})
            
            # Use .get() to safely access keys (handles error dicts)
            vol = ratio.get('volume_pcr')
            oi = ratio.get('oi_pcr')
            
            # Format as "Vol: X.XX / OI: Y.YY" or "N/A" if failed
            if vol is not None and oi is not None:
                vols.append(vol)
                ois.append(oi)
                sector_results.append(f"{vol:.2f} / {oi:.2f}")
            else:
                sector_results.append("N/A")
        
        # Compute sector average (across successful providers only)
        if vols:
            avg_vol = sum(vols) / len(vols)
            avg_oi = sum(ois) / len(ois)
            avg_str = f"{avg_vol:.2f} / {avg_oi:.2f}"
        else:
            avg_str = "N/A"
        sector_results.append(avg_str)  # Add average as fourth item
        
        table_data[sector] = sector_results 
    
    # Intuitive output: Markdown table (rows: sectors, columns: providers)
    print("\n" + "=" * 90)
    print("PUT/CALL RATIOS BY SECTOR AND PROVIDER")
    print("(Volume PCR / OI PCR; N/A if fetch failed; Average across successful providers)")
    print("-" * 90)
    
    # Header
    header = "| Sector\t\t| " + " | ".join([f"{prov:^9}" for prov in providers]) + " |"
    print(header)
    print("|" + "-" * 22 + "|" + ("-" * 11) * len(providers) + "|")
    
    # Rows
    for sector, values in table_data.items():
        row = f"| {sector:<20} | " + " | ".join([f"{val:^9}" for val in values]) + " |"
        print(row)
    
    print("-" * 90)
    
    print(f"\nFinal cache stats: {get_cache_stats()}")

if __name__ == "__main__":
    # test_optimized_features()
    test_putcall_ratio()
    # ratio = get_put_call_ratio("SPY")
    # print(f"Volume P/C Ratio: {ratio['volume_pcr']:.2f}")
    # print(f"OI P/C Ratio: {ratio['oi_pcr']:.2f}")
    