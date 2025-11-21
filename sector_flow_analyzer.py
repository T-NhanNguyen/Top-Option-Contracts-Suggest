# SECTOR MONEY FLOW ANALYSIS - PYTHON IMPLEMENTATION
# Uses yfinance, pandas, and standard libraries
# No complex dependencies needed

"""
This script implements the Layer 2 (Sector Money Flow) analysis from our framework.
It provides real-time sector performance analysis vs SPY with volume confirmation.

Requirements:
    pip install yfinance pandas numpy requests

Usage:
    from sector_flow import SectorFlowAnalyzer
    
    analyzer = SectorFlowAnalyzer()
    result = analyzer.analyze_stock("BE")
    print(result)
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')


class SectorFlowAnalyzer:
    """
    Analyzes sector money flow for stock market trading decisions.
    
    Provides Layer 2 analysis: Sector performance vs SPY with volume confirmation.
    """
    
    # Sector ETF mapping
    SECTOR_ETFS = {
        'Technology': 'XLK',
        'Energy': 'XLE',
        'Financials': 'XLF',
        'Healthcare': 'XLV',
        'Industrials': 'XLI',
        'Consumer Discretionary': 'XLY',
        'Consumer Staples': 'XLP',
        'Materials': 'XLB',
        'Real Estate': 'XLRE',
        'Utilities': 'XLU',
        'Communication Services': 'XLC'
    }
    
    def __init__(self):
        """Initialize the analyzer."""
        self.cache = {}  # Cache for reducing API calls
        
    def get_stock_sector(self, ticker: str) -> Optional[str]:
        """
        Get the sector for a given stock ticker.
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL', 'BE')
            
        Returns:
            Sector name or None if not found
        """
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            # Try different field names Yahoo Finance uses
            sector = info.get('sector') or info.get('sectorKey')
            
            if sector:
                return sector
            else:
                print(f"⚠️ Warning: Could not find sector for {ticker}")
                return None
                
        except Exception as e:
            print(f"❌ Error getting sector for {ticker}: {e}")
            return None
    
    def get_sector_etf(self, sector: str) -> Optional[str]:
        """
        Get the corresponding ETF ticker for a sector.
        
        Args:
            sector: Sector name
            
        Returns:
            ETF ticker symbol
        """
        return self.SECTOR_ETFS.get(sector)
    
    def get_intraday_data(self, ticker: str, period: str = '1d', interval: str = '1m') -> pd.DataFrame:
        """
        Get intraday data for a ticker.
        
        Args:
            ticker: Stock or ETF symbol
            period: Data period ('1d', '5d', etc.)
            interval: Data interval ('1m', '5m', '1h')
            
        Returns:
            DataFrame with OHLCV data
        """
        try:
            data = yf.download(ticker, period=period, interval=interval, progress=False)
            return data
        except Exception as e:
            print(f"❌ Error downloading data for {ticker}: {e}")
            return pd.DataFrame()
    
    def get_daily_performance(self, ticker: str) -> Tuple[float, float, float]:
        """
        Get today's performance metrics for a ticker.
        
        Args:
            ticker: Stock or ETF symbol
            
        Returns:
            Tuple of (current_price, change_percent, volume_ratio)
        """
        try:
            # Get intraday data
            data = self.get_intraday_data(ticker, period='5d', interval='5m')
            
            if data.empty:
                return 0.0, 0.0, 0.0
            
            # Get today's data
            today = data.last('1D')
            
            if len(today) == 0:
                return 0.0, 0.0, 0.0
            
            # Calculate metrics
            open_price = today['Open'].iloc[0]
            current_price = today['Close'].iloc[-1]
            change_percent = ((current_price - open_price) / open_price) * 100
            
            # Volume analysis - compare today vs recent average
            # Get last 20 days of data for volume comparison
            hist_data = yf.download(ticker, period='1mo', interval='1d', progress=False)
            
            if not hist_data.empty and len(hist_data) >= 5:
                today_volume = today['Volume'].sum()
                avg_volume = hist_data['Volume'].tail(20).mean()
                volume_ratio = today_volume / avg_volume if avg_volume > 0 else 1.0
            else:
                volume_ratio = 1.0
            
            return current_price, change_percent, volume_ratio
            
        except Exception as e:
            print(f"❌ Error calculating performance for {ticker}: {e}")
            return 0.0, 0.0, 0.0
    
    def analyze_sector_flow(self, sector: str) -> Dict:
        """
        Analyze money flow for a specific sector.
        
        Args:
            sector: Sector name
            
        Returns:
            Dictionary with sector flow analysis
        """
        sector_etf = self.get_sector_etf(sector)
        
        if not sector_etf:
            return {
                'sector': sector,
                'etf': None,
                'error': 'No ETF mapping found'
            }
        
        # Get sector performance
        sector_price, sector_change, sector_vol_ratio = self.get_daily_performance(sector_etf)
        
        # Get SPY performance for comparison
        spy_price, spy_change, spy_vol_ratio = self.get_daily_performance('SPY')
        
        # Calculate relative performance
        relative_performance = sector_change - spy_change
        
        # Determine flow status based on our framework
        if relative_performance > 0.5 and sector_vol_ratio > 1.2:
            flow_status = "STRONG_INFLOW"
            score = 1.0
            emoji = "✅"
        elif relative_performance > 0 and sector_vol_ratio > 1.0:
            flow_status = "MODERATE_INFLOW"
            score = 0.8
            emoji = "✅"
        elif -0.5 <= relative_performance <= 0:
            flow_status = "NEUTRAL"
            score = 0.5
            emoji = "⚠️"
        elif relative_performance <= -0.5 and sector_vol_ratio < 1.5:
            flow_status = "WEAK_OUTFLOW"
            score = 0.3
            emoji = "⚠️"
        else:
            flow_status = "STRONG_OUTFLOW"
            score = 0.0
            emoji = "🚨"
        
        return {
            'sector': sector,
            'etf': sector_etf,
            'sector_change': sector_change,
            'spy_change': spy_change,
            'relative_performance': relative_performance,
            'volume_ratio': sector_vol_ratio,
            'flow_status': flow_status,
            'score': score,
            'emoji': emoji,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def analyze_stock(self, ticker: str, verbose: bool = True) -> Dict:
        """
        Complete analysis for a stock including sector flow.
        
        Args:
            ticker: Stock symbol
            verbose: Print detailed output
            
        Returns:
            Dictionary with complete analysis
        """
        if verbose:
            print(f"\n{'='*60}")
            print(f"ANALYZING: {ticker}")
            print(f"{'='*60}\n")
        
        # Step 1: Get stock's sector
        sector = self.get_stock_sector(ticker)
        
        if not sector:
            return {
                'ticker': ticker,
                'error': 'Could not determine sector',
                'decision': 'SKIP'
            }
        
        if verbose:
            print(f"Stock: {ticker}")
            print(f"Sector: {sector}")
        
        # Step 2: Analyze sector flow
        sector_analysis = self.analyze_sector_flow(sector)
        
        if verbose:
            print(f"\n--- SECTOR FLOW ANALYSIS ---")
            print(f"Sector ETF: {sector_analysis['etf']}")
            print(f"Sector Performance: {sector_analysis['sector_change']:.2f}%")
            print(f"SPY Performance: {sector_analysis['spy_change']:.2f}%")
            print(f"Relative Performance: {sector_analysis['relative_performance']:.2f}%")
            print(f"Volume Ratio: {sector_analysis['volume_ratio']:.2f}x")
            print(f"\n{sector_analysis['emoji']} Flow Status: {sector_analysis['flow_status']}")
            print(f"Layer 2 Score: {sector_analysis['score']:.2f}/1.0")
        
        # Step 3: Analyze individual stock
        stock_price, stock_change, stock_vol_ratio = self.get_daily_performance(ticker)
        
        # Calculate stock vs sector relative strength
        stock_vs_sector = stock_change - sector_analysis['sector_change']
        
        if verbose:
            print(f"\n--- STOCK ANALYSIS ---")
            print(f"Stock Performance: {stock_change:.2f}%")
            print(f"Stock Volume Ratio: {stock_vol_ratio:.2f}x")
            print(f"Relative to Sector: {stock_vs_sector:+.2f}%")
        
        # Determine relative strength status
        if stock_vs_sector > 1.0:
            rel_strength = "OUTPERFORMING"
            rel_emoji = "✅"
        elif stock_vs_sector > 0:
            rel_strength = "ALIGNED"
            rel_emoji = "✅"
        elif stock_vs_sector > -1.0:
            rel_strength = "SLIGHT_LAG"
            rel_emoji = "⚠️"
        else:
            rel_strength = "UNDERPERFORMING"
            rel_emoji = "🚨"
        
        if verbose:
            print(f"{rel_emoji} Relative Strength: {rel_strength}")
        
        # Step 4: Make decision
        if sector_analysis['score'] < 0.3:
            decision = "DO_NOT_OPEN"
            reason = "Sector outflows - money rotating away"
        elif stock_vs_sector < -1.0 and stock_vol_ratio > 1.5:
            decision = "DO_NOT_OPEN"
            reason = "Stock underperforming sector significantly"
        elif sector_analysis['score'] >= 0.8:
            decision = "FAVORABLE"
            reason = "Strong sector inflows + good stock behavior"
        elif sector_analysis['score'] >= 0.5:
            decision = "NEUTRAL"
            reason = "Sector neutral - wait for clarity"
        else:
            decision = "CAUTION"
            reason = "Weak sector flows - reduce position size"
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"DECISION: {decision}")
            print(f"Reason: {reason}")
            print(f"{'='*60}\n")
        
        return {
            'ticker': ticker,
            'sector': sector,
            'sector_analysis': sector_analysis,
            'stock_performance': stock_change,
            'stock_volume_ratio': stock_vol_ratio,
            'relative_strength': stock_vs_sector,
            'relative_strength_status': rel_strength,
            'decision': decision,
            'reason': reason,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
    
    def analyze_multiple_stocks(self, tickers: List[str]) -> pd.DataFrame:
        """
        Analyze multiple stocks and return comparison DataFrame.
        
        Args:
            tickers: List of stock symbols
            
        Returns:
            DataFrame with analysis for all stocks
        """
        results = []
        
        for ticker in tickers:
            print(f"\nAnalyzing {ticker}...")
            analysis = self.analyze_stock(ticker, verbose=False)
            
            if 'error' not in analysis:
                results.append({
                    'Ticker': ticker,
                    'Sector': analysis['sector'],
                    'Stock %': f"{analysis['stock_performance']:.2f}",
                    'Vol Ratio': f"{analysis['stock_volume_ratio']:.2f}x",
                    'Rel Strength': f"{analysis['relative_strength']:+.2f}%",
                    'Sector Flow': analysis['sector_analysis']['flow_status'],
                    'Score': f"{analysis['sector_analysis']['score']:.2f}",
                    'Decision': analysis['decision']
                })
        
        df = pd.DataFrame(results)
        return df
    
    def get_sector_heatmap(self) -> pd.DataFrame:
        """
        Get a heatmap of all sector performances vs SPY.
        
        Returns:
            DataFrame with sector performance metrics
        """
        print("\n" + "="*60)
        print("SECTOR HEATMAP vs SPY")
        print("="*60 + "\n")
        
        results = []
        
        # Get SPY performance once
        spy_price, spy_change, spy_vol_ratio = self.get_daily_performance('SPY')
        
        print(f"SPY Performance: {spy_change:.2f}%\n")
        
        for sector, etf in self.SECTOR_ETFS.items():
            analysis = self.analyze_sector_flow(sector)
            
            results.append({
                'Sector': sector,
                'ETF': etf,
                'Change %': f"{analysis['sector_change']:.2f}%",
                'vs SPY': f"{analysis['relative_performance']:+.2f}%",
                'Vol Ratio': f"{analysis['volume_ratio']:.2f}x",
                'Flow': analysis['flow_status'],
                'Status': analysis['emoji'],
                'Score': f"{analysis['score']:.2f}"
            })
        
        df = pd.DataFrame(results)
        
        # Sort by relative performance
        df = df.sort_values(by='vs SPY', ascending=False, key=lambda x: x.str.replace('%', '').str.replace('+', '').astype(float))
        
        return df


# ============================================
# STANDALONE USAGE EXAMPLES
# ============================================

def example_single_stock():
    """Example: Analyze a single stock."""
    analyzer = SectorFlowAnalyzer()
    
    # Analyze BE (Bloom Energy)
    result = analyzer.analyze_stock("BE")
    
    return result


def example_multiple_stocks():
    """Example: Compare multiple stocks in your portfolio."""
    analyzer = SectorFlowAnalyzer()
    
    # Your AI portfolio
    portfolio = ['BE', 'FLNC', 'IREN', 'NBIS', 'AMD', 'NVDA']
    
    df = analyzer.analyze_multiple_stocks(portfolio)
    print("\n" + "="*60)
    print("PORTFOLIO ANALYSIS")
    print("="*60)
    print(df.to_string(index=False))
    
    return df


def example_sector_heatmap():
    """Example: Get full sector heatmap."""
    analyzer = SectorFlowAnalyzer()
    
    df = analyzer.get_sector_heatmap()
    print(df.to_string(index=False))
    
    return df


def monday_morning_routine(portfolio: List[str]):
    """
    Complete Monday morning analysis routine.
    
    Args:
        portfolio: List of stock tickers in your portfolio
    """
    print("\n" + "="*70)
    print("MONDAY MORNING SECTOR FLOW ANALYSIS")
    print("="*70)
    
    analyzer = SectorFlowAnalyzer()
    
    # Step 1: Get sector heatmap
    print("\n📊 STEP 1: Sector Heatmap")
    sector_df = analyzer.get_sector_heatmap()
    print(sector_df.to_string(index=False))
    
    # Step 2: Analyze your portfolio
    print("\n📈 STEP 2: Your Portfolio Analysis")
    portfolio_df = analyzer.analyze_multiple_stocks(portfolio)
    print("\n" + portfolio_df.to_string(index=False))
    
    # Step 3: Recommendations
    print("\n💡 STEP 3: Recommendations")
    print("="*70)
    
    for _, row in portfolio_df.iterrows():
        ticker = row['Ticker']
        decision = row['Decision']
        
        if decision == "FAVORABLE":
            print(f"✅ {ticker}: Green light - sector flowing in, good for opening/adding")
        elif decision == "NEUTRAL":
            print(f"⚠️  {ticker}: Yellow light - hold existing, avoid new positions")
        elif decision == "CAUTION":
            print(f"⚠️  {ticker}: Caution - consider reducing position size")
        else:
            print(f"🚨 {ticker}: Red light - trim position or avoid opening")
    
    print("="*70 + "\n")
    
    return sector_df, portfolio_df


# ============================================
# MAIN EXECUTION
# ============================================

if __name__ == "__main__":
    # Example 1: Single stock analysis
    print("\n" + "="*70)
    print("EXAMPLE 1: Single Stock Analysis")
    print("="*70)
    example_single_stock()
    
    # Example 2: Multiple stocks comparison
    print("\n" + "="*70)
    print("EXAMPLE 2: Portfolio Comparison")
    print("="*70)
    example_multiple_stocks()
    
    # Example 3: Sector heatmap
    print("\n" + "="*70)
    print("EXAMPLE 3: Sector Heatmap")
    print("="*70)
    example_sector_heatmap()
    
    # Example 4: Full Monday morning routine
    print("\n" + "="*70)
    print("EXAMPLE 4: Monday Morning Routine")
    print("="*70)
    
    # Your AI-concentrated portfolio
    my_portfolio = ['BE', 'FLNC', 'IREN', 'NBIS', 'AMD', 'NVDA']
    
    sector_heatmap, portfolio_analysis = monday_morning_routine(my_portfolio)