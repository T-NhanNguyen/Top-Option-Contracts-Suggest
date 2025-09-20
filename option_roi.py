import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List
from gamma_calculator import calculate_gamma, calculate_delta

GAMMA_LOW_THRESHOLD = 0.02
GAMMA_HIGH_THRESHOLD = 0.05
DEBUG_MODE = False

class OptionROIAnalyzer:
    def __init__(self):
        self.risk_free_rate = 0.05
        self.percentile_threshold = 0.8  # Top 20% for percentile-based filtering

    def screen_by_time_adjusted_gamma(self, options_df: pd.DataFrame, strategy: str, use_percentile: bool = False) -> pd.DataFrame:
        """Screen options based on Time-Adjusted Gamma or percentile-based filtering."""
        options_df['time_adjusted_gamma'] = options_df['gamma'] * np.sqrt(options_df['dte'] / 365)
        if use_percentile:
            gamma_threshold = options_df['time_adjusted_gamma'].quantile(self.percentile_threshold)
        else:
            gamma_threshold = GAMMA_LOW_THRESHOLD if strategy == "undervalued" else GAMMA_HIGH_THRESHOLD
        return options_df[options_df['time_adjusted_gamma'] > gamma_threshold].copy()

    def rank_by_gamma_per_dollar(self, options_df: pd.DataFrame) -> pd.DataFrame:
        """Rank options by Gamma per Dollar."""
        options_df['gamma_per_dollar'] = options_df['gamma'] / options_df['last_price']
        return options_df.nlargest(10, 'gamma_per_dollar')

    def calculate_taylor_adjusted_roi(self, options_df: pd.DataFrame, current_price: float, target_price: float,
                                     investment_amount: float, strategy: str, forecast_move_percent: float = None) -> pd.DataFrame:
        """
        Calculate Taylor Series Gamma-Adjusted ROI with proper mathematical formulation.
        """
        result_df = options_df.copy()
        
        for idx, option in result_df.iterrows():
            try:
                # Calculate expected stock price move ΔS
                if strategy == "undervalued":
                    if forecast_move_percent is None:
                        forecast_move_percent = option['iv'] * np.sqrt(option['dte'] / 365)
                    delta_s = current_price * forecast_move_percent
                    expected_price = current_price + delta_s
                else:  # catalyst
                    delta_s = target_price - current_price
                    expected_price = target_price
                
                # Calculate expected option price change using Taylor expansion
                price_change = (option['delta'] * delta_s) + (0.5 * option['gamma'] * (delta_s ** 2))
                
                # For puts, adjust the sign
                if option['type'] == 'put':
                    price_change = price_change
                
                # Calculate expected option price
                expected_option_price = option['last_price'] + price_change
                expected_option_price = max(expected_option_price, 0.01)
                
                # Calculate ROI
                contracts_can_buy = investment_amount // (option['last_price'] * 100)
                if contracts_can_buy == 0:
                    continue
                
                potential_profit = (expected_option_price - option['last_price']) * 100 * contracts_can_buy
                roi_percentage = (potential_profit / investment_amount) * 100
                
                # Store results
                result_df.loc[idx, 'expected_stock_move'] = delta_s
                result_df.loc[idx, 'expected_option_price'] = expected_option_price
                result_df.loc[idx, 'price_change_taylor'] = price_change
                result_df.loc[idx, 'potential_profit'] = potential_profit
                result_df.loc[idx, 'taylor_roi'] = roi_percentage
                result_df.loc[idx, 'contracts_affordable'] = contracts_can_buy
                
            except (ValueError, ZeroDivisionError, KeyError) as e:
                print(f"Error calculating Taylor ROI for option {option['strike']}: {e}")
                continue
        
        return result_df.nlargest(10, 'taylor_roi')

    def calculate_black_scholes(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str) -> float:
        """Calculate Black-Scholes option price."""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        if option_type == 'call':
            return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    def analyze_options(self, options_data: List[Dict], current_price: float, strategy: str,
                        investment_amount: float, target_price: float = None, forecast_move: float = 0.02) -> Dict:
        """Analyze options and return ranked opportunities."""
        options_df = pd.DataFrame(options_data)
        if options_df.empty:
            return {"error": "No options data provided"}

        # Calculate Greeks
        options_df['gamma'] = options_df.apply(
            lambda x: calculate_gamma(current_price, x['strike'], x['dte'] / 365, self.risk_free_rate, x['iv']), axis=1)
        options_df['delta'] = options_df.apply(
            lambda x: calculate_delta(current_price, x['strike'], x['dte'] / 365, x['iv'], x['type']), axis=1)

        # Screen by Time-Adjusted Gamma (try both methods, prefer threshold for undervalued, percentile for catalyst)
        use_percentile = strategy == "catalyst"
        
        # To test and validate calculations
        if DEBUG_MODE == False:
            filtered_df = self.screen_by_time_adjusted_gamma(options_df, strategy, use_percentile)
        else:
            filtered_df = options_df.copy()  # Use all options

        if filtered_df.empty:
            return {"error": "No contracts passed Gamma screening"}

        # Rank by Gamma per Dollar for undervalued, or by ROI for catalyst
        if strategy == "undervalued":
            ranked_df = self.rank_by_gamma_per_dollar(filtered_df)
        else:
            ranked_df = self.calculate_taylor_adjusted_roi(
                filtered_df.copy(), current_price, target_price, investment_amount, strategy, forecast_move)

        # Final selection (top 5) with execution filter
        final_selection = ranked_df[
            (ranked_df['last_price'] > 0) &
            (ranked_df['oi'] >= 500) &
            (ranked_df['volume'] >= 100)
        ].head(5)

        return {
            'best_opportunities': final_selection.to_dict('records'),
            'all_opportunities': filtered_df.to_dict('records')
        }
    
# Example Usage
if __name__ == "__main__":
    try:
        analyzer = OptionROIAnalyzer()
        options_data = [
            {
                'strike': 150.0,  # ATM
                'dte': 30,
                'iv': 0.5,
                'last_price': 5.0,
                'type': 'call',
                'oi': 1000,
                'volume': 200
            },
            {
                'strike': 155.0,  # OTM
                'dte': 30,
                'iv': 0.6,
                'last_price': 3.0,
                'type': 'call',
                'oi': 800,
                'volume': 150
            },
            {
                'strike': 145.0, 
                'dte': 30, 
                'iv': 0.5, 
                'last_price': 4.0, 
                'type': 'put', 
                'oi': 900, 
                'volume': 180
            }
        ]
        result = analyzer.analyze_options(
            options_data=options_data,
            current_price=150.0,
            strategy='catalyst',
            investment_amount=5000,
            target_price=172.5
        )
        if 'error' in result:
            print(f"Error in analyze_options: {result['error']}")
        else:
            print("Test results:")
            for opt in result['best_opportunities']:
                print(f"Strike: {opt['strike']}, Taylor ROI: {opt['taylor_roi']:.2f}%, Delta: {opt['delta']:.6f}")
            delta = result['best_opportunities'][0]['delta'] if result['best_opportunities'] else None
            print(f"\nanalyze_options test - Delta: {delta:.6f}")
    except Exception as e:
        print(f"Error in analyze_options test: {e}")