import pandas as pd
from datetime import datetime, timedelta
from option_chain import get_option_chain_analysis_optimized, clear_cache, get_cache_stats, get_put_call_ratio
from gamma_calculator import gamma_calculator, calculate_gamma, calculate_delta
from option_roi import OptionROIAnalyzer
import ascii_art_print
import numpy as np

import argparse



def find_highest_roi_options(
    ticker: str,
    strategy: str = "undervalued",
    min_dte: int = 30,
    max_dte: int = 210,
    investment_amount: float = 10000,
    min_volume: int = 100,
    min_oi: int = 500,
    target_price_multiplier: float = 1.15,
    use_taylor_series: bool = True,
    forecast_move_percent: float = None
):
    """
    Find highest ROI options based on specified strategy with optional Taylor series adjustment.

    Parameters:
    ticker (str): Stock ticker symbol
    strategy (str): Analysis mode ("undervalued" or "catalyst")
    min_dte (int): Minimum days to expiration (default: 30)
    max_dte (int): Maximum days to expiration (default: 90, ignored for undervalued strategy)
    investment_amount (float): Amount to invest for ROI calculation
    min_volume (int): Minimum volume filter for liquidity
    min_oi (int): Minimum open interest filter for liquidity
    target_price_multiplier (float): Target stock price as a multiple of current price
    use_taylor_series (bool): Whether to use Taylor series for ROI calculation
    forecast_move_percent (float): Expected move percentage for Taylor series (e.g., 0.02 for 2%)

    Returns:
    dict: Analysis results with top ROI opportunities
    """
    if strategy not in ["undervalued", "catalyst"]:
        return {"error": "Invalid strategy. Choose 'undervalued' or 'catalyst'"}
    
    # Print helper functions take cares of this
    # print(f"Analyzing {ticker} for {strategy} strategy with DTE >= {min_dte}...")
    # if strategy == "catalyst":
    #     if forecast_move_percent is not None:
    #         print(f"DTE range: {min_dte}-{max_dte}, Target price movement: {forecast_move_percent}")
    #     else:
    #         print(f"DTE range: {min_dte}-{max_dte}, Target price multiplier: {target_price_multiplier}")
    # if use_taylor_series:
    #     print(f"Using Taylor series ROI calculation with {'default' if forecast_move_percent is None else f'{forecast_move_percent:.1%}'} move")
    # print("-" * 60)
    
    # Get option chain analysis
    analysis = get_option_chain_analysis_optimized(ticker, min_dte)
    
    if "error" in analysis:
        print(f"Error: {analysis['error']}")
        return analysis
    
    current_price = analysis['current_price']
    target_price = None
    if forecast_move_percent is not None:
        # Use forecast_move_percent if provided
        target_price = current_price + (current_price * forecast_move_percent)
    else:
        # Fall back to target_price_multiplier if forecast_move_percent is None
        target_price = current_price * target_price_multiplier
    
    # print helper function takes care of this
    # print(f"Current price: ${current_price:.2f}" + 
    #       (f" | Target price: ${target_price:.2f}" if strategy == "catalyst" else ""))
    # print(f"Found {analysis['qualified_expirations_count']} qualified expirations")
    # print(f"Total OI: {analysis['total_open_interest']:,}")
    # print(f"Total Volume: {analysis['total_volume']:,}")
    
    # Extract strikes based on strategy
    all_strikes = []
    strike_details = {}
    today = datetime.now()
    for expiry, data in analysis['expiration_data'].items():
        expiry_date = datetime.strptime(expiry, '%Y-%m-%d')
        dte = (expiry_date - today).days
        # print(f"  {expiry}: DTE = {dte}")
        
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
                
                # Check if required fields exist in the option data
                required_fields = ['impliedVolatility', 'openInterest', 'volume', 'lastPrice']
                if not all(field in option for field in required_fields):
                    print(f"Skipping option {strike} {opt_type}: missing required fields")
                    continue
                    
                all_strikes.append(strike)
                strike_details[strike] = {
                    'type': opt_type[:-1],
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

    # Calculate gamma for all strikes with actual DTE for each option
    gamma_values = []
    for strike in unique_strikes:
        if strike in strike_details:
            details = strike_details[strike]
            dte_years = details['dte'] / 365.0
            iv_decimal = details['iv']  # Assuming this is already in decimal form
            
            # Add validation checks before gamma calculation
            if (current_price <= 0 or 
                dte_years <= 0 or 
                iv_decimal <= 0 or 
                pd.isna(iv_decimal)):
                # print(f"Skipping option {option['strike']}: invalid inputs - S: {current_price}, T: {dte_years}, sigma: {iv_decimal}")
                continue

            gamma = calculate_gamma(
                S=current_price,
                K=strike,
                T=dte_years,
                sigma=iv_decimal,
                dte_days=details['dte']
            )
            gamma_values.append(gamma)
        else:
            gamma_values.append(0)  # Default value if strike not found

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

    # Check if required columns exist for filtering
    required_columns = ['volume', 'oi']
    if not all(col in strike_df.columns for col in required_columns):
        missing_cols = [col for col in required_columns if col not in strike_df.columns]
        print(f"Warning: Missing columns in strike_df: {missing_cols}. Available: {strike_df.columns.tolist()}")
        return {"error": f"Missing required columns for liquidity filtering: {missing_cols}"}

    # Filter for liquid options
    liquid_options = strike_df[
        (strike_df['volume'] >= min_volume) & 
        (strike_df['oi'] >= min_oi)
    ]

    
    if liquid_options.empty:
        return {"error": f"No liquid {'OTM call' if strategy == 'catalyst' else 'valid'} options found meeting criteria"}
    
    # Calculate ROI based on strategy
    analyzer = OptionROIAnalyzer()
    analyzer.risk_free_rate = 0.05
    
    roi_results = []

    for _, option in liquid_options.iterrows():
        try:
            # Validate IV to avoid unrealistic values
            if option['iv'] > 1.1:  # IV > 100%
                print(f"Warning: Extreme IV ({option['iv']:.2%}) for {option['type']} ${option['strike']} (DTE: {option['dte']})")
                continue

            # Check if dte exists and is a valid number
            if 'dte' not in option or not isinstance(option['dte'], (int, float)) or pd.isna(option['dte']) or option['dte'] <= 0:
                print(f"Skipping option with invalid DTE: {option.get('strike', 'unknown')}, DTE: {option.get('dte')}")
                continue

            dte_years = option['dte'] / 365.0
            if dte_years <= 0:  # Skip if DTE is zero or negative
                continue

            delta = calculate_delta(
                S=current_price,
                K=option['strike'],
                T=dte_years,
                sigma=option['iv'],
                option_type=option['type'],
                dte_days=option['dte']
            )

            # Calculate option price based on strategy
            if strategy == "undervalued":
                calc_price = analyzer.calculate_black_scholes(
                    S=current_price,
                    K=option['strike'],
                    T=dte_years,
                    sigma=option['iv'],
                    option_type=option['type'],
                    dte_days=option['dte']
                )
            else:  # catalyst
                calc_price = analyzer.calculate_black_scholes(
                    S=target_price,
                    K=option['strike'],
                    T=dte_years,
                    sigma=option['iv'],
                    option_type='call',
                    dte_days=option['dte']
                )
            
            # ROI calculation
            contracts_can_buy = investment_amount // (option['last_price'] * 100)
            if contracts_can_buy == 0:
                continue
            
            potential_profit = (calc_price - option['last_price']) * 100 * contracts_can_buy
            roi_percentage = (potential_profit / investment_amount) * 100
            
            # Skip negative ROIs for all results
            # if roi_percentage <= 0:
            #     continue
                
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
    if use_taylor_series:
        best_roi = roi_df.nlargest(10, 'roi_percentage')
    else:
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
        'use_taylor_series': use_taylor_series,
        'forecast_move_percent': forecast_move_percent,
        'best_roi_opportunities': best_roi.to_dict('records'),
        secondary_label: secondary_opportunities,
        'all_opportunities': roi_df.to_dict('records')
    }

def analyze_cross_strategy_opportunities(undervalued_results, catalyst_results):
    """
    Identify and analyze contracts that appear in both undervalued and catalyst strategies
    Provides insights on pricing efficiency and catalyst potential
    """
    if "error" in undervalued_results or "error" in catalyst_results:
        return {"error": "Cannot compare - one or both strategies failed"}
    
    # Create dictionaries for quick lookup
    undervalued_map = {}
    for opp in undervalued_results.get('all_opportunities', []):
        key = f"{opp['type']}_{opp['strike']}_{opp['expiration']}"
        undervalued_map[key] = opp
    
    catalyst_map = {}
    for opp in catalyst_results.get('all_opportunities', []):
        key = f"{opp['type']}_{opp['strike']}_{opp['expiration']}"
        catalyst_map[key] = opp
    
    # Find common contracts
    common_contracts = []
    for key in set(undervalued_map.keys()) & set(catalyst_map.keys()):
        undervalued_data = undervalued_map[key]
        catalyst_data = catalyst_map[key]
        
        common_contracts.append({
            'contract_key': key,
            'strike': undervalued_data['strike'],
            'type': undervalued_data['type'],
            'expiration': undervalued_data['expiration'],
            'dte': undervalued_data['dte'],
            'market_price': undervalued_data['market_price'],
            'undervalued_roi': undervalued_data['roi_percentage'],
            'catalyst_roi': catalyst_data['roi_percentage'],
            'iv': undervalued_data['iv'],
            'gamma': undervalued_data['gamma'],
            'delta': undervalued_data['delta'],
            'price_discrepancy': undervalued_data.get('price_discrepancy', 0),
            'efficiency_ratio': abs(catalyst_data['roi_percentage'] / max(0.01, undervalued_data['roi_percentage']))
        })
    
    # Sort by most interesting opportunities (high catalyst ROI with reasonable undervalued ROI)
    common_contracts.sort(key=lambda x: (
        -x['catalyst_roi'],  # Highest catalyst ROI first
        x['efficiency_ratio']  # Then by efficiency ratio
    ))
    
    return {
        'common_contracts': common_contracts,
        'total_common': len(common_contracts),
        'insights': generate_strategy_insights(common_contracts)
    }

def generate_strategy_insights(common_contracts):
    """Generate actionable insights from cross-strategy analysis"""
    insights = []
    
    for contract in common_contracts[:5]:  # Top 5 most interesting
        uv_roi = contract['undervalued_roi']
        cat_roi = contract['catalyst_roi']
        efficiency = contract['efficiency_ratio']
        
        if abs(uv_roi) < 5 and cat_roi > 50:
            insight = (
                f"${contract['strike']} {contract['type']} ({contract['dte']} DTE): "
                f"Efficiently priced ({uv_roi:.1f}% ROI) with explosive catalyst potential ({cat_roi:.1f}% ROI). "
                f"Great risk/reward if catalyst thesis plays out."
            )
        elif uv_roi < -10 and cat_roi > 100:
            insight = (
                f"${contract['strike']} {contract['type']}: "
                f"Overpriced ({uv_roi:.1f}% ROI) but massive catalyst upside ({cat_roi:.1f}% ROI). "
                f"Consider selling instead of buying, or use spreads."
            )
        elif uv_roi > 10 and cat_roi > 75:
            insight = (
                f"${contract['strike']} {contract['type']}: "
                f"Undervalued ({uv_roi:.1f}% ROI) with strong catalyst potential ({cat_roi:.1f}% ROI). "
                f"Excellent opportunity - both strategies align."
            )
        else:
            insight = (
                f"${contract['strike']} {contract['type']}: "
                f"UV ROI: {uv_roi:.1f}%, Catalyst ROI: {cat_roi:.1f}%. "
                f"Efficiency Ratio: {efficiency:.1f}x"
            )
        
        insights.append(insight)
    
    return insights

def parse_date_range(date_range_str: str) -> tuple:
    """
    Parse a simple date range format like '11/25-1/26' or '11/25-11/30'
    Returns (start_date, end_date) as datetime objects

    Format: MM/DD-MM/DD
    Assumes current year for start date, and handles year rollover
    """
    try:
        parts = date_range_str.split('-')
        if len(parts) != 2:
            raise ValueError("Date range must be in format MM/DD-MM/DD")

        start_str, end_str = parts

        # Parse month/day
        start_month, start_day = map(int, start_str.split('/'))
        end_month, end_day = map(int, end_str.split('/'))

        # Get current year
        current_year = datetime.now().year

        # Create start date with current year
        start_date = datetime(current_year, start_month, start_day)

        # If start date is in the past, use next year
        if start_date < datetime.now():
            start_date = datetime(current_year + 1, start_month, start_day)

        # Create end date
        # If end month is less than start month, assume next year
        if end_month < start_month:
            end_date = datetime(start_date.year + 1, end_month, end_day)
        else:
            end_date = datetime(start_date.year, end_month, end_day)

        return start_date, end_date

    except Exception as e:
        raise ValueError(f"Error parsing date range '{date_range_str}': {str(e)}")

def get_filtered_options_by_date(
    ticker: str,
    date_range: str,
    min_volume: int = 100,
    min_oi: int = 500,
    option_type: str = "calls",  # "calls", "puts", or "both"
    include_greeks: bool = False
) -> dict:
    """
    Get option contracts within a specified date range, filtered by OI and volume.

    Parameters:
    ticker (str): Stock ticker symbol
    date_range (str): Date range in format 'MM/DD-MM/DD' (e.g., '11/25-1/26')
    min_volume (int): Minimum volume filter (default: 100)
    min_oi (int): Minimum open interest filter (default: 500)
    option_type (str): "calls", "puts", or "both" (default: "calls")
    include_greeks (bool): Whether to calculate and include Greeks (delta, gamma)

    Returns:
    dict: Contains filtered options and metadata
    """
    try:
        # Parse the date range
        start_date, end_date = parse_date_range(date_range)

        # Get option chain analysis (min_dte=0 to get all available options)
        analysis = get_option_chain_analysis_optimized(ticker, min_dte=0)

        if "error" in analysis:
            return {"error": analysis["error"]}

        current_price = analysis['current_price']
        today = datetime.now()

        # Collect options within the date range
        filtered_options = []

        for expiry, data in analysis['expiration_data'].items():
            expiry_date = datetime.strptime(expiry, '%Y-%m-%d')

            # Filter by date range
            if not (start_date <= expiry_date <= end_date):
                continue

            dte = (expiry_date - today).days

            # Determine which option types to process
            types_to_process = []
            if option_type in ["calls", "both"]:
                types_to_process.append(('calls', 'call'))
            if option_type in ["puts", "both"]:
                types_to_process.append(('puts', 'put'))

            for opt_type_key, opt_type_label in types_to_process:
                for option in data[opt_type_key]:
                    # Apply volume and OI filters
                    if option.get('volume', 0) < min_volume:
                        continue
                    if option.get('openInterest', 0) < min_oi:
                        continue

                    strike = option['strike']
                    contract_id = option.get('contractSymbol', '')

                    option_data = {
                        'contract_id': contract_id,
                        'strike': strike,
                        'type': opt_type_label,
                        'expiration': expiry,
                        'dte': dte,
                        'last_price': option.get('lastPrice', 0),
                        'bid': option.get('bid', 0),
                        'ask': option.get('ask', 0),
                        'iv': option.get('impliedVolatility', 0),
                        'oi': option.get('openInterest', 0),
                        'volume': option.get('volume', 0)
                    }

                    # Calculate Greeks if requested
                    if include_greeks and dte > 0:
                        dte_years = dte / 365.0
                        iv_decimal = option.get('impliedVolatility', 0)

                        if current_price > 0 and dte_years > 0 and iv_decimal > 0:
                            try:
                                gamma = calculate_gamma(
                                    S=current_price,
                                    K=strike,
                                    T=dte_years,
                                    sigma=iv_decimal,
                                    dte_days=dte
                                )
                                delta = calculate_delta(
                                    S=current_price,
                                    K=strike,
                                    T=dte_years,
                                    sigma=iv_decimal,
                                    option_type=opt_type_label,
                                    dte_days=dte
                                )
                                option_data['gamma'] = gamma
                                option_data['delta'] = delta
                            except:
                                option_data['gamma'] = 0
                                option_data['delta'] = 0

                    filtered_options.append(option_data)

        if not filtered_options:
            return {
                "error": f"No options found for {ticker} in date range {date_range} with volume>={min_volume} and OI>={min_oi}"
            }

        # Sort by expiration date, then by strike
        filtered_options.sort(key=lambda x: (x['expiration'], x['strike']))

        return {
            'ticker': ticker,
            'current_price': current_price,
            'date_range': date_range,
            'start_date': start_date.strftime('%Y-%m-%d'),
            'end_date': end_date.strftime('%Y-%m-%d'),
            'filters': {
                'min_volume': min_volume,
                'min_oi': min_oi,
                'option_type': option_type
            },
            'options_count': len(filtered_options),
            'options': filtered_options
        }

    except Exception as e:
        return {"error": f"Error filtering options: {str(e)}"}

def format_options_for_chat(results: dict, include_greeks: bool = False) -> str:
    """
    Format filtered options results for easy pasting into chat.

    Parameters:
    results (dict): Results from get_filtered_options_by_date()
    include_greeks (bool): Whether to include Greeks in output

    Returns:
    str: Formatted string ready for chat
    """
    if "error" in results:
        return f"Error: {results['error']}"

    output = []
    output.append(f"📊 {results['ticker']} Option Contracts")
    output.append(f"Current Price: ${results['current_price']:.2f}")
    output.append(f"Date Range: {results['start_date']} to {results['end_date']}")
    output.append(f"Filters: Volume≥{results['filters']['min_volume']}, OI≥{results['filters']['min_oi']}")
    output.append(f"Found {results['options_count']} contracts")
    output.append("")

    # Group by expiration date
    options_by_expiry = {}
    for opt in results['options']:
        expiry = opt['expiration']
        if expiry not in options_by_expiry:
            options_by_expiry[expiry] = []
        options_by_expiry[expiry].append(opt)

    # Format each expiration group
    for expiry in sorted(options_by_expiry.keys()):
        options = options_by_expiry[expiry]
        dte = options[0]['dte']

        output.append(f"📅 {expiry} (DTE: {dte})")
        output.append("-" * 80)

        # Header
        if include_greeks:
            output.append(f"{'Contract ID':<25} {'Bid':<7} {'Ask':<7} {'Last':<7} {'IV':<7} {'OI':<9} {'Vol':<7} {'Delta':<7} {'Gamma':<7}")
        else:
            output.append(f"{'Contract ID':<25} {'Bid':<7} {'Ask':<7} {'Last':<7} {'IV':<7} {'OI':<9} {'Vol':<7}")

        # Options
        for opt in sorted(options, key=lambda x: x['strike']):
            contract_id = opt.get('contract_id', f"{opt['type'].upper()}_{opt['strike']}")

            if include_greeks and 'delta' in opt and 'gamma' in opt:
                output.append(
                    f"{contract_id:<25} ${opt['bid']:<6.2f} ${opt['ask']:<6.2f} ${opt['last_price']:<6.2f} "
                    f"{opt['iv']:<6.1%} {int(opt['oi']):<9,} {int(opt['volume']):<7,} "
                    f"{opt['delta']:<7.3f} {opt['gamma']:<7.4f}"
                )
            else:
                output.append(
                    f"{contract_id:<25} ${opt['bid']:<6.2f} ${opt['ask']:<6.2f} ${opt['last_price']:<6.2f} "
                    f"{opt['iv']:<6.1%} {int(opt['oi']):<9,} {int(opt['volume']):<7,}"
                )

        output.append("")

    return "\n".join(output)

def print_results(undervalued_results, catalyst_results, detailed=False):
    """Print formatted results for both strategies with the new design"""
    # Header and ROI comparison
    
    # Get analysis date from either result
    analysis_date = undervalued_results.get('analysis_date', datetime.now().strftime('%Y-%m-%d'))
    if "error" in undervalued_results and "error" in catalyst_results:
        print("Error: Both strategies failed")
        return
    
    # Get current price from whichever result is available
    current_price = None
    if "error" not in undervalued_results:
        current_price = undervalued_results['current_price']
    elif "error" not in catalyst_results:
        current_price = catalyst_results['current_price']
    
    investment_amount = 5000  # Default or get from results
    
    print(f"{analysis_date} | Price: ${current_price:.2f} | Buying Power: ${investment_amount:,.0f}")
    
    strategy_info = []
    if "error" not in undervalued_results:
        move_percent = undervalued_results.get('forecast_move_percent', 0.5)
        strategy_info.append(f"Undervalued (DTE ≥ {undervalued_results['min_dte']}, {move_percent:.0%} Move)")
    
    if "error" not in catalyst_results:
        target_info = f"Target ${catalyst_results['target_price']:.2f}" if catalyst_results.get('target_price') else "No Target"
        max_dte = catalyst_results.get('max_dte', 90)
        strategy_info.append(f"Catalyst (DTE {catalyst_results['min_dte']}-{max_dte}, {target_info})")
    
    print(" | ".join(strategy_info))
    print()
    
    # ROI Comparison chart - combine opportunities from both strategies
    print("ROI% Comparison (Undervalued vs. Catalyst)")
    print()
    
    # Get top opportunities from both strategies
    uv_opportunities = undervalued_results.get('best_roi_opportunities', []) if "error" not in undervalued_results else []
    cat_opportunities = catalyst_results.get('best_roi_opportunities', []) if "error" not in catalyst_results else []
    
    # Combine and sort by ROI
    all_opportunities = []
    for opp in uv_opportunities:
        opp['strategy'] = 'undervalued'
        all_opportunities.append(opp)
    
    for opp in cat_opportunities:
        opp['strategy'] = 'catalyst'
        all_opportunities.append(opp)
    
    # Sort by ROI descending and take top 5
    all_opportunities.sort(key=lambda x: x['roi_percentage'], reverse=True)
    top_opportunities = all_opportunities[:5]
    
    # Determine scale and range
    if top_opportunities:
        max_roi = max(opp['roi_percentage'] for opp in top_opportunities)
        min_roi = min(opp['roi_percentage'] for opp in top_opportunities)
        roi_range = max(max_roi, abs(min_roi))
        scale_unit = max(20, roi_range / 20)  # At least 20% per block, or scale dynamically
    else:
        scale_unit = 20
        roi_range = 100
    
    for opp in top_opportunities:
        roi = opp['roi_percentage']
        bar_length = max(1, int(abs(roi) / scale_unit))
        bar = "█" * bar_length if roi > 0 else "░" * bar_length
        
        option_type = "Call" if opp['type'] == 'call' else "Put"
        strategy_label = "Underval" if opp['strategy'] == 'undervalued' else "Catalyst"
        
        print(f"{option_type} {opp['strike']:.1f}\t({strategy_label})\t|{bar} {roi:.1f}%")
    
    print()
    print(f"Scale: █ = ~{scale_unit:.0f}% ROI | Range: {-roi_range:.0f}% to {roi_range:.0f}%")
    print("—" * 60)  
        
    # Combine all opportunities from both strategies
    uv_all = undervalued_results.get('all_opportunities', []) if "error" not in undervalued_results else []
    cat_all = catalyst_results.get('all_opportunities', []) if "error" not in catalyst_results else []
    
    all_combined = uv_all + cat_all
    
    if all_combined:
        # Sort by ROI descending
        all_combined.sort(key=lambda x: x['roi_percentage'], reverse=True)
        
        if detailed:
            print("Type Strike Exp      \tDTE ROI% \tIV%  \tPrice\t\tOI/Vol\t\tGamma        Delta")
            print("---- ------ -------  \t--- ---- \t---- \t--------\t--------\t-----        -----")
        else:
            print("Type Strike Exp      \tDTE ROI% \tIV%  \tPrice\t\tOI/Vol")
            print("---- ------ -------  \t--- ---- \t---- \t--------\t--------")

            
        for opp in all_combined[:10]:  # Show top 10
                option_type = "Call" if opp['type'] == 'call' else "Put"
                exp_date = opp['expiration'].split('-')
                exp_formatted = f"{exp_date[1]}-{exp_date[2]}-{exp_date[0][2:]}"
                price_info = f"{opp['market_price']:.2f}/{opp.get('calc_price', 0):.2f}"
                oi_vol = f"{int(opp['oi']):,}/{int(opp['volume']):,}"
                
                if detailed:
                    gamma = f"{opp.get('gamma', 0):.4f}" if 'gamma' in opp else "N/A"
                    delta = f"{opp.get('delta', 0):.4f}" if 'delta' in opp else "N/A"
                    print(f"{option_type:4} {opp['strike']:6.1f} {exp_formatted:9} \t{opp['dte']:3} {opp['roi_percentage']:5.1f}% \t{opp['iv']:4.1%} \t{price_info:9} \t{oi_vol:9} \t{gamma:7}        {delta:7}")
                else:
                    print(f"{option_type:4} {opp['strike']:6.1f} {exp_formatted:9} \t{opp['dte']:3} {opp['roi_percentage']:5.1f}% \t{opp['iv']:4.1%} \t{price_info:9} \t{oi_vol:9}")    
    else:
        print("No opportunities found")
    
    print("—" * 60)


def print_cross_strategy_analysis(analysis, detailed=False):
    """Print the cross-strategy analysis results with the new design"""
    if "error" in analysis:
        print(f"Cross-strategy analysis error: {analysis['error']}")
        return
    
    # Cross-strategy analysis
    print("CROSS-STRATEGY ANALYSIS: Contracts in Both Undervalued & Catalyst")
    print()
    
    common_contracts = analysis.get('common_contracts', [])
    total_common = analysis.get('total_common', 0)
    ascii_art_print.print_wizard()
    print(f"Found {total_common} contract{'s' if total_common != 1 else ''} appearing in both strategies!")
    print()
    
    
    print("Most Interesting Opportunities")
    insights = analysis.get('insights', [])
    if insights:
        for i, insight in enumerate(insights, 1):
            print(f"{i}. {insight}")
    else:
        print("No common opportunities found")
    print()
    
    if detailed and common_contracts:
        print("Detailed Comparison")
        print("Contract\t\t| UV ROI\t| Catalyst ROI\t| Efficiency\t| IV\t| Gamma")
        print("-" * 80)
        
        for contract in common_contracts[:10]:
            option_type = "call" if contract['type'] == 'call' else "put"
            exp_date = contract['expiration'].split('-')
            exp_formatted = f"{exp_date[1]}-{exp_date[2]}-{exp_date[0][2:]}"
            
            print(f"{option_type} ${contract['strike']:5.1f} {exp_formatted}\t| "
                  f"{contract['undervalued_roi']:6.1f}%\t| "
                  f"{contract['catalyst_roi']:11.1f}%\t| "
                  f"{contract['efficiency_ratio']:9.1f}x\t| "
                  f"{contract['iv']:4.1%}\t| "
                  f"{contract['gamma']:6.4f}")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Options Analysis Tool',
        epilog='Examples:\n'
               '  Full analysis:     python main.py AAPL --taylor\n'
               '  Date filter:       python main.py UAMY --date-range 11/25-1/26\n'
               '  With Greeks:       python main.py UAMY --date-range 11/25-1/26 --greeks\n'
               '  Custom filters:    python main.py UAMY --date-range 12/1-12/31 --min-volume 50 --min-oi 100',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Required arguments
    parser.add_argument('ticker', help='Stock ticker symbol')

    # Date filter mode arguments
    parser.add_argument('--date-range', type=str,
                       help='Filter options by date range (format: MM/DD-MM/DD, e.g., 11/25-1/26)')
    parser.add_argument('--option-type', choices=['calls', 'puts', 'both'], default='calls',
                       help='Option type to filter (default: calls)')
    parser.add_argument('--greeks', action='store_true',
                       help='Include Greeks (delta, gamma) in date filter output')

    # Common filter arguments
    parser.add_argument('--min-volume', type=int, default=100,
                       help='Minimum volume filter (default: 100)')
    parser.add_argument('--min-oi', type=int, default=500,
                       help='Minimum open interest filter (default: 500)')

    # Full analysis mode arguments
    parser.add_argument('--taylor', action='store_true', help='Use Taylor series ROI calculation')
    parser.add_argument('--move', type=float, help='Expected move percentage (e.g., 0.02 for 2%%)')
    parser.add_argument('--detailed', action='store_true', help='Show detailed tables')
    parser.add_argument('--strategy', choices=['undervalued', 'catalyst'], default='undervalued',
                       help='Analysis strategy')
    parser.add_argument('--min-dte', type=int, default=30, help='Minimum days to expiration')
    parser.add_argument('--investment', type=float, default=10000, help='Investment amount')

    args = parser.parse_args()

    ticker = args.ticker.upper()

    # Clear any previous cache
    clear_cache()

    # Check if date filter mode is requested
    if args.date_range:
        # Date filter mode - use the new helper functions
        print(f"Filtering {ticker} options by date range: {args.date_range}")
        print(f"Filters: Volume ≥ {args.min_volume}, OI ≥ {args.min_oi}, Type: {args.option_type}")
        print("=" * 60)

        results = get_filtered_options_by_date(
            ticker=ticker,
            date_range=args.date_range,
            min_volume=args.min_volume,
            min_oi=args.min_oi,
            option_type=args.option_type,
            include_greeks=args.greeks
        )

        if "error" in results:
            print(f"\nError: {results['error']}")
        else:
            # Format and print results
            formatted = format_options_for_chat(results, include_greeks=args.greeks)
            print(formatted)

            # Show summary
            print("\n" + "=" * 60)
            print(f"Summary: Found {results['options_count']} contracts")
            print("Ready to copy and paste into chat!")
    else:
        # Full analysis mode - original functionality
        use_taylor = args.taylor
        forecast_move = args.move
        detailed = args.detailed

        # Analyze a stock with both strategies
        ascii_art_print.print_wizard_message(f"Running analysis for {ticker}...")

        results_undervalued = find_highest_roi_options(
            ticker=ticker,
            strategy="undervalued",
            min_dte=45,
            investment_amount=5000,
            min_volume=args.min_volume,
            min_oi=args.min_oi,
            use_taylor_series=use_taylor,
            forecast_move_percent=forecast_move
        )

        results_catalyst = find_highest_roi_options(
            ticker=ticker,
            strategy="catalyst",
            min_dte=30,
            max_dte=90,
            investment_amount=5000,
            min_volume=args.min_volume,
            min_oi=args.min_oi,
            target_price_multiplier=1.5,  # 50% move for catalyst
            use_taylor_series=use_taylor,
            forecast_move_percent=forecast_move
        )

        # Print combined results
        print_results(results_undervalued, results_catalyst, detailed)

        # print("\n" + "=" * 80 + "\n")
        cross_analysis = analyze_cross_strategy_opportunities(results_undervalued, results_catalyst)
        print_cross_strategy_analysis(cross_analysis, detailed)

        # Show cache statistics
        print("\nCache Statistics:")
        print(get_cache_stats())

# Example usage of the helper functions:
#
# To get filtered options by date range:
# results = get_filtered_options_by_date(
#     ticker="UAMY",
#     date_range="11/25-1/26",  # November 25 to January 26
#     min_volume=100,            # Minimum volume
#     min_oi=500,                # Minimum open interest
#     option_type="calls",       # "calls", "puts", or "both"
#     include_greeks=True        # Include delta and gamma calculations
# )
#
# To format the results for chat:
# formatted_output = format_options_for_chat(results, include_greeks=True)
# print(formatted_output)
#
# Complete example:
# if __name__ == "__main__":
#     clear_cache()
#     results = get_filtered_options_by_date("UAMY", "11/25-1/26", min_volume=50, min_oi=100)
#     print(format_options_for_chat(results))