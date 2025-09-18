import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, List, Tuple
import yfinance as yf
from datetime import datetime

class OptionROIAnalyzer:
    """Complete system to find highest ROI option contracts using Black-Scholes"""
    
    def __init__(self):
        self.risk_free_rate = 0.05  # Default, can be updated
    
    def fetch_option_chain(self, ticker: str, expiration: str = None) -> pd.DataFrame:
        """Fetch live option chain data from Yahoo Finance"""
        stock = yf.Ticker(ticker)
        opt_chain = stock.option_chain(expiration)
        
        # Combine calls and puts
        calls = opt_chain.calls
        puts = opt_chain.puts
        calls['type'] = 'call'
        puts['type'] = 'put'
        
        option_chain = pd.concat([calls, puts])
        return option_chain
    
    def calculate_black_scholes(self, S: float, K: float, T: float, r: float, 
                              sigma: float, option_type: str) -> float:
        """Calculate Black-Scholes option price"""
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        
        if option_type == 'call':
            price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        
        return price
    
    def calculate_gamma(self, S: float, K: float, T: float, sigma: float) -> float:
        """Calculate Gamma for risk assessment"""
        d1 = (np.log(S / K) + (self.risk_free_rate + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        pdf_d1 = norm.pdf(d1)
        return pdf_d1 / (S * sigma * np.sqrt(T))
    
    def calculate_delta(self, S: float, K: float, T: float, sigma: float, option_type: str) -> float:
        """Calculate Delta for directional exposure"""
        d1 = (np.log(S / K) + (self.risk_free_rate + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        if option_type == 'call':
            return norm.cdf(d1)
        else:  # put
            return norm.cdf(d1) - 1
    
    def calculate_roi_metrics(self, option_chain: pd.DataFrame, S: float, 
                            days_to_expiry: int, investment_amount: float = 10000) -> pd.DataFrame:
        """Calculate comprehensive ROI metrics for all options"""
        
        results = []
        T = days_to_expiry / 365
        
        for _, option in option_chain.iterrows():
            try:
                # Skip options with missing data
                if pd.isna(option['impliedVolatility']) or option['lastPrice'] == 0:
                    continue
                
                # Theoretical price
                theoretical_price = self.calculate_black_scholes(
                    S, option['strike'], T, self.risk_free_rate, 
                    option['impliedVolatility'], option['type']
                )
                
                # Current market price
                market_price = option['lastPrice']
                
                # ROI calculation (simplified)
                contracts_can_buy = investment_amount // (market_price * 100)
                if contracts_can_buy == 0:
                    continue
                
                # Potential profit if price moves to theoretical value
                potential_profit = (theoretical_price - market_price) * 100 * contracts_can_buy
                roi_percentage = (potential_profit / investment_amount) * 100
                
                # Greeks for risk assessment
                gamma = self.calculate_gamma(S, option['strike'], T, option['impliedVolatility'])
                delta = self.calculate_delta(S, option['strike'], T, option['impliedVolatility'], option['type'])
                
                results.append({
                    'strike': option['strike'],
                    'type': option['type'],
                    'market_price': market_price,
                    'theoretical_price': theoretical_price,
                    'implied_volatility': option['impliedVolatility'],
                    'price_discrepancy': theoretical_price - market_price,
                    'roi_percentage': roi_percentage,
                    'gamma': gamma,
                    'delta': delta,
                    'open_interest': option['openInterest'],
                    'volume': option['volume'],
                    'contracts_affordable': contracts_can_buy,
                    'potential_profit': potential_profit
                })
                
            except (ValueError, ZeroDivisionError):
                continue
        
        return pd.DataFrame(results)
    
    def find_best_opportunities(self, ticker: str, investment_amount: float = 10000, 
                               min_volume: int = 100, min_oi: int = 500) -> Dict:
        """Main function to find highest ROI opportunities"""
        
        # Fetch current data
        stock = yf.Ticker(ticker)
        S = stock.info['currentPrice']
        option_chain = self.fetch_option_chain(ticker)
        
        # Get nearest expiration
        expirations = stock.options
        if not expirations:
            raise ValueError("No option expirations available")
        
        nearest_expiry = expirations[0]
        expiry_date = datetime.strptime(nearest_expiry, '%Y-%m-%d')
        days_to_expiry = (expiry_date - datetime.now()).days
        
        # Calculate ROI metrics
        roi_df = self.calculate_roi_metrics(option_chain, S, days_to_expiry, investment_amount)
        
        # Filter for liquid options
        liquid_options = roi_df[
            (roi_df['volume'] >= min_volume) & 
            (roi_df['open_interest'] >= min_oi)
        ]
        
        if liquid_options.empty:
            return {"error": "No liquid options found meeting criteria"}
        
        # Find best opportunities by different metrics
        best_roi = liquid_options.nlargest(5, 'roi_percentage')
        best_undervalued = liquid_options.nlargest(5, 'price_discrepancy')
        best_low_risk = liquid_options[
            (liquid_options['gamma'].abs() < 0.1) &  # Low gamma = less price sensitivity
            (liquid_options['roi_percentage'] > 5)
        ].nlargest(5, 'roi_percentage')
        
        return {
            'current_price': S,
            'days_to_expiry': days_to_expiry,
            'best_roi_opportunities': best_roi.to_dict('records'),
            'best_undervalued': best_undervalued.to_dict('records'),
            'best_low_risk': best_low_risk.to_dict('records'),
            'all_opportunities': liquid_options.to_dict('records')
        }

# Usage Example
if __name__ == "__main__":
    analyzer = OptionROIAnalyzer()
    
    # Analyze Tesla options for highest ROI
    results = analyzer.find_best_opportunities('TSLA', investment_amount=5000)
    
    print(f"Current Price: ${results['current_price']:.2f}")
    print(f"Days to Expiry: {results['days_to_expiry']}")
    print("\nTop 5 ROI Opportunities:")
    for opportunity in results['best_roi_opportunities']:
        print(f"{opportunity['type'].upper()} ${opportunity['strike']}: "
              f"{opportunity['roi_percentage']:.1f}% ROI, "
              f"Gamma: {opportunity['gamma']:.4f}")