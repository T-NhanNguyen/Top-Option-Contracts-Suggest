import pandas as pd
from datetime import datetime
from option_chain import get_option_chain_analysis_optimized, clear_cache, get_cache_stats
from gamma_calculator import gamma_calculator, calculate_gamma
from option_roi import OptionROIAnalyzer
import numpy as np
import sys

def find_highest_roi_options(
    ticker: str,
    strategy: str = "undervalued",
    min_dte: int = 30,
    max_dte: int = 210,
    investment_amount: float = 10000,
    min_volume: int = 100,
    min_oi: int = 500,
    target_price_multiplier: float = 1.15
):
    """
    Find highest ROI options based on specified strategy.

    Parameters:
    ticker (str): Stock ticker symbol
    strategy (str): Analysis mode ("undervalued" or "catalyst")
    min_dte (int): Minimum days to expiration (default: 30)
    max_dte (int): Maximum days to expiration (default: 90, ignored for undervalued strategy)
    investment_amount (float): Amount to invest for ROI calculation
    min_volume (int): Minimum volume filter for liquidity
    min_oi (int): Minimum open interest filter for liquidity
    target_price_multiplier (float): Target stock price as a multiple of current price (default: 1.15, used in catalyst strategy)

    Returns:
    dict: Analysis results with top ROI opportunities
    """
    if strategy not in ["undervalued", "catalyst"]:
        return {"error": "Invalid strategy. Choose 'undervalued' or 'catalyst'"}
    
    print(f"Analyzing {ticker} for {strategy} strategy with DTE >= {min_dte}...")
    if strategy == "catalyst":
        print(f"DTE range: {min_dte}-{max_dte}, Target price multiplier: {target_price_multiplier}")
    print("=" * 60)
    
    # Get option chain analysis
    analysis = get_option_chain_analysis_optimized(ticker, min_dte)
    
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return analysis
    
    current_price = analysis['current_price']
    target_price = current_price * target_price_multiplier if strategy == "catalyst" else None
    print(f"Current price: ${current_price:.2f}" + 
          (f" | Target price: ${target_price:.2f}" if strategy == "catalyst" else ""))
    print(f"Found {analysis['qualified_expirations_count']} qualified expirations")
    print(f"Total OI: {analysis['total_open_interest']:,}")
    print(f"Total Volume: {analysis['total_volume']:,}")
    
    # Extract strikes based on strategy
    all_strikes = []
    strike_details = {}
    
    for expiry, data in analysis['expiration_data'].items():
        expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
        dte = (expiry_date - datetime.now()).days
        
        # For catalyst strategy, skip expirations beyond max_dte
        if strategy == "catalyst" and dte > max_dte:
            continue
        
        # Process options based on strategy
        option_types = ['calls'] if strategy == "catalyst" else ['calls', 'puts']
        for opt_type in option_types:
            for option in data[opt_type]:
                strike = option['strike']
                # For catalyst, only include OTM calls
                if strategy == "catalyst" and (opt_type != 'calls' or strike <= current_price):
                    continue
                all_strikes.append(strike)
                strike_details[strike] = {
                    'type': opt_type[:-1],  # 'calls' -> 'call', 'puts' -> 'put'
                    'expiration': expiry,
                    'dte': dte,
                    'iv': option['impliedVolatility'],
                    'oi': option['openInterest'],
                    'volume': option['volume'],
                    'last_price': option['lastPrice']
                }
    
    # Remove duplicates and sort
    unique_strikes = sorted(list(set(all_strikes)))
    
    if not unique_strikes:
        return {"error": f"No {'OTM call' if strategy == 'catalyst' else 'valid'} options found meeting criteria"}
    
    # Calculate gamma for all strikes
    # print("\nCalculating gamma exposure for strikes...")
    gamma_values = gamma_calculator.calculate_gamma_vectorized(
        S=current_price,
        strikes=np.array(unique_strikes),
        T=min_dte/365,  # Conservative estimate
        r=0.05,
        sigma=analysis.get('average_implied_volatility', 0.3)
    )
    
    # Create strike analysis DataFrame
    strike_analysis = []
    for strike, gamma in zip(unique_strikes, gamma_values):
        if strike in strike_details:
            details = strike_details[strike]
            strike_analysis.append({
                'strike': strike,
                'type': details['type'],
                'expiration': details['expiration'],
                'dte': details['dte'],
                'iv': details['iv'],
                'oi': details['oi'],
                'volume': details['volume'],
                'last_price': details['last_price'],
                'gamma': gamma
            })
    
    strike_df = pd.DataFrame(strike_analysis)
    
    # Filter for liquid options
    liquid_options = strike_df[
        (strike_df['volume'] >= min_volume) & 
        (strike_df['oi'] >= min_oi)
    ]
    
    if liquid_options.empty:
        return {"error": f"No liquid {'OTM call' if strategy == 'catalyst' else 'valid'} options found meeting criteria"}
    
    # Calculate ROI based on strategy
    # print(f"Calculating ROI for liquid options ({strategy} strategy)...")
    analyzer = OptionROIAnalyzer()
    analyzer.risk_free_rate = 0.05
    
    roi_results = []
    
    for _, option in liquid_options.iterrows():
        try:
            # Validate IV to avoid unrealistic values
            if option['iv'] > 1.1:  # IV > 100%
                print(f"Warning: Extreme IV ({option['iv']:.2%}) for {option['type']} ${option['strike']} (DTE: {option['dte']})")
                continue

            # Calculate option price based on strategy
            if strategy == "undervalued":
                calc_price = analyzer.calculate_black_scholes(
                    S=current_price,
                    K=option['strike'],
                    T=option['dte']/365,
                    r=analyzer.risk_free_rate,
                    sigma=option['iv'],
                    option_type=option['type']
                )
            else:  # catalyst
                calc_price = analyzer.calculate_black_scholes(
                    S=target_price,
                    K=option['strike'],
                    T=option['dte']/365,
                    r=analyzer.risk_free_rate,
                    sigma=option['iv'],
                    option_type='call'
                )
            
            # ROI calculation
            contracts_can_buy = investment_amount // (option['last_price'] * 100)
            if contracts_can_buy == 0:
                continue
            
            potential_profit = (calc_price - option['last_price']) * 100 * contracts_can_buy
            roi_percentage = (potential_profit / investment_amount) * 100
            
            # Skip negative ROIs for all results
            if roi_percentage <= 0:
                continue

            # Calculate delta
            delta = analyzer.calculate_delta(
                S=current_price,
                K=option['strike'],
                T=option['dte']/365,
                sigma=option['iv'],
                option_type=option['type']
            )
            
            roi_results.append({
                'strike': option['strike'],
                'type': option['type'],
                'expiration': option['expiration'],
                'dte': option['dte'],
                'market_price': option['last_price'],
                'calc_price': calc_price,
                'price_discrepancy': calc_price - option['last_price'],
                'roi_percentage': roi_percentage,
                'gamma': option['gamma'],
                'delta': delta,
                'iv': option['iv'],
                'oi': option['oi'],
                'volume': option['volume'],
                'contracts_affordable': contracts_can_buy,
                'potential_profit': potential_profit
            })
            
        except (ValueError, ZeroDivisionError):
            continue
    
    if not roi_results:
        return {"error": "No valid ROI calculations could be performed"}
    
    roi_df = pd.DataFrame(roi_results)
    
    # Find best opportunities
    best_roi = roi_df[roi_df['roi_percentage'] > 0].nlargest(10, 'roi_percentage')
    
    # Additional filters based on strategy
    if strategy == "undervalued":
        low_risk = roi_df[
            (roi_df['gamma'].abs() < 0.1) & 
            (roi_df['roi_percentage'] > 5)
        ].nlargest(10, 'roi_percentage')
        secondary_opportunities = low_risk.to_dict('records')
        secondary_label = "best_low_risk"
    else:  # catalyst
        high_gamma = roi_df[
            (roi_df['gamma'] > 0.05) & 
            (roi_df['roi_percentage'] > 50)
        ].nlargest(10, 'roi_percentage')
        secondary_opportunities = high_gamma.to_dict('records')
        secondary_label = "high_gamma_opportunities"
    
    return {
        'ticker': ticker,
        'current_price': current_price,
        'target_price': target_price if strategy == "catalyst" else None,
        'analysis_date': datetime.now().strftime('%Y-%m-%d'),
        'min_dte': min_dte,
        'max_dte': max_dte if strategy == "catalyst" else None,
        'investment_amount': investment_amount,
        'strategy': strategy,
        'best_roi_opportunities': best_roi.to_dict('records'),
        secondary_label: secondary_opportunities,
        'all_opportunities': roi_df.to_dict('records')
    }

def print_results(results):
    """Print formatted results from the analysis in a clean card-style format"""
    if "error" in results:
        print(f"Error: {results['error']}")
        return
    
    # Wrap output in a markdown code block for Discord/.md compatibility
    # print("```markdown")
    
    # Strategy header
    # strategy_label = "Calculated Underpriced Options" if results['strategy'] == "undervalued" else "Catalyst Strategy"
    # print(f"**{results['ticker']} {strategy_label}**")
    print(f"\nSearch Date: {results['analysis_date']}")
    print(f"Current Price: ${results['current_price']:.2f}" + 
          (f" | Target: ${results['target_price']:.2f}" if results['strategy'] == 'catalyst' else ""))
    print(f"Buying Power: ${results['investment_amount']:,.0f}")
    print(f"DTE: {results['min_dte']}" + 
          (f"-{results['max_dte']}" if results['strategy'] == 'catalyst' else "+") + " days")
    print()
    
    # Top ROI picks
    print("**Top ROI Picks**")
    top_picks = results.get('best_roi_opportunities', [])[:5]
    if not top_picks:
        print("    Calculation didn't yield any worthy picks")
    else:
        for opp in top_picks:
            if not all(key in opp for key in ['type', 'strike', 'expiration', 'dte', 'roi_percentage', 'iv', 'market_price']):
                continue
            print(f"  {opp['type'].title()} ${opp['strike']:.1f} | Exp: {opp['expiration']} | DTE: {opp['dte']}")
            print(f"    ROI: {opp['roi_percentage']:.1f}% | IV: {opp['iv']:.2%}")
            print(f"    Market: ${opp['market_price']:.2f} | " + 
                  (f"Target: ${opp.get('calc_price', 0):.2f}" if results.get('strategy') == 'catalyst' else 
                   f"Theoretical: ${opp.get('calc_price', 0):.2f}"))
            print(f"    Gamma: {(opp.get('gamma', 0) * 1000):.4f}e-3 | Delta: {(opp.get('delta', 0) * 1000):.4f}e-3")
            print(f"    OI: {opp.get('oi', 0):,} | Vol: {opp.get('volume', 0):,}")
            print()
    
    # Secondary picks
    secondary_label = "Conservative Picks" if results.get('strategy') == "undervalued" else "High-Gamma Picks"
    secondary_key = 'best_low_risk' if results.get('strategy') == "undervalued" else 'high_gamma_opportunities'
    print(f"**{secondary_label}**")
    secondary_picks = results.get(secondary_key, [])[:5]
    if not secondary_picks:
        print("    Calculation didn't yield any worthy picks")
    else:
        for opp in secondary_picks:
            if not all(key in opp for key in ['type', 'strike', 'expiration', 'dte', 'roi_percentage', 'iv', 'market_price']):
                continue
            print(f"  {opp['type'].title()} ${opp['strike']:.1f} | Exp: {opp['expiration']} | DTE: {opp['dte']}")
            print(f"    ROI: {opp['roi_percentage']:.1f}% | IV: {opp['iv']:.2%}")
            print(f"    Market: ${opp['market_price']:.2f} | " + 
                  (f"Target: ${opp.get('calc_price', 0):.2f}" if results.get('strategy') == 'catalyst' else 
                   f"Theoretical: ${opp.get('calc_price', 0):.2f}"))
            print(f"    Gamma: {(opp.get('gamma', 0) * 1000):.4f}e-3 | Delta: {(opp.get('delta', 0) * 1000):.4f}e-3")
            print(f"    OI: {opp.get('oi', 0):,} | Vol: {opp.get('volume', 0):,}")
            print()
    
    # print("```")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 main.py [ticker]")
        sys.exit(1)
    # Clear any previous cache
    clear_cache()
    
    # Analyze a stock with both strategies
    ticker = sys.argv[1].upper()
    
    # print("Running UNDERVALUED strategy...")
    results_undervalued = find_highest_roi_options(
        ticker=ticker,
        strategy="undervalued",
        min_dte=45,
        investment_amount=5000,
        min_volume=100,
        min_oi=500
    )
    print_results(results_undervalued)
    
    print("\n" + "=" * 80 + "\n")
    # print("Running CATALYST strategy...")
    results_catalyst = find_highest_roi_options(
        ticker=ticker,
        strategy="catalyst",
        min_dte=30,
        investment_amount=5000,
        min_volume=100,
        min_oi=500,
        target_price_multiplier=1.5
    )
    print_results(results_catalyst)
    
    # Show cache statistics
    print("\nCache Statistics:")
    print(get_cache_stats())
