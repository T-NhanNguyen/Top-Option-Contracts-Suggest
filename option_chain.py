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
                    'calls': chain.calls[['strike', 'openInterest', 'impliedVolatility', 'lastPrice', 'volume', 'bid', 'ask']],
                    'puts': chain.puts[['strike', 'openInterest', 'impliedVolatility', 'lastPrice', 'volume', 'bid', 'ask']]
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
    expiry_date = datetime.strptime(expiration, '%Y-%m-%d').date()
    dte = (expiry_date - today.date()).days
    
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
        required_cols = ['strike', 'openInterest', 'impliedVolatility', 'lastPrice', 'volume', 'bid', 'ask']
        for col in required_cols:
            if col not in df.columns:
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
if __name__ == "__main__":
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