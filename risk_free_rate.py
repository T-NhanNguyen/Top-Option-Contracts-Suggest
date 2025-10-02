import yfinance as yf
from datetime import datetime, timedelta
import threading

class RiskFreeRateManager:
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        with cls._lock:
            if cls._instance is None:
                cls._instance = super(RiskFreeRateManager, cls).__new__(cls)
                cls._instance._initialize()
            return cls._instance
    
    def _initialize(self):
        self._short_term_rate = None
        self._long_term_rate = None
        self._last_update = None
        self._cache_duration = timedelta(hours=4)  # Refresh every 4 hours
    
    def _needs_refresh(self):
        if self._last_update is None:
            return True
        return datetime.now() - self._last_update > self._cache_duration
    
    def _fetch_rates(self):
        try:
            # Fetch short-term rate (3-month T-bill)
            irx_data = yf.Ticker("^IRX").history(period="1d")
            self._short_term_rate = irx_data['Close'].iloc[-1] / 100
            
            # Fetch long-term rate (10-year Treasury)
            tnx_data = yf.Ticker("^TNX").history(period="1d")
            self._long_term_rate = tnx_data['Close'].iloc[-1] / 100
            
            self._last_update = datetime.now()
            
        except Exception as e:
            print(f"Warning: Failed to fetch risk-free rates, using defaults. Error: {e}")
            # Set reasonable defaults if API fails
            if self._short_term_rate is None:
                self._short_term_rate = 0.05
            if self._long_term_rate is None:
                self._long_term_rate = 0.04
    
    def get_rate(self, dte_days: int) -> float:
        """Get risk-free rate based on days to expiration."""
        with self._lock:
            if self._needs_refresh():
                self._fetch_rates()
            
            time_horizon_years = dte_days / 365.0
            
            if time_horizon_years <= 1:
                rate = self._short_term_rate
            else:
                rate = self._long_term_rate
            
            # Ensure rate is reasonable
            # print(f"Calculated Risk Free rates are {rate}")
            return max(0.001, min(rate, 0.15))

# Global instance for easy access
risk_free_manager = RiskFreeRateManager()

# Convenience function
def get_risk_free_rate(dte_days: int) -> float:
    return risk_free_manager.get_rate(dte_days)