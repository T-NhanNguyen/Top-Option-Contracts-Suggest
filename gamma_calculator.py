import numpy as np
from scipy.stats import norm
from typing import Union, List, Dict

class BlackScholesGammaCalculator:
    """Simplified Black-Scholes Gamma calculator with vectorized operations."""

    def __init__(self):
        # No caching by default, as NumPy operations are efficient
        pass

    def _validate_inputs(self, S: float, K: Union[float, np.ndarray], T: float, r: float, sigma: float) -> None:
        """Validate input parameters with vector-aware checks."""
        if S <= 0 or T <= 0 or sigma <= 0:
            raise ValueError("S, T, and sigma must be positive, non-zero values.")
        
        if isinstance(K, np.ndarray):
            if np.any(K <= 0):
                raise ValueError("All strike prices in K must be positive, non-zero values.")
        elif K <= 0:
            raise ValueError("Strike price K must be positive, non-zero value.")

    def _calculate_d1_vectorized(self, S: float, K: Union[float, np.ndarray], T: float, r: float, sigma: float) -> np.ndarray:
        """Vectorized calculation of d1 for single or multiple strike prices."""
        K = np.asarray(K)  # Ensure K is a NumPy array for vectorized operations
        return (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

    def calculate_gamma(self, S: float, K: float, T: float, r: float, sigma: float) -> float:
        """
        Calculate Gamma for a single option.

        Parameters:
        S (float): Current stock price
        K (float): Strike price
        T (float): Time to expiration (in years)
        r (float): Risk-free rate
        sigma (float): Implied volatility

        Returns:
        float: Gamma value
        """
        self._validate_inputs(S, K, T, r, sigma)
        d1 = self._calculate_d1_vectorized(S, K, T, r, sigma)
        pdf_d1 = norm.pdf(d1)
        return pdf_d1 / (S * sigma * np.sqrt(T))

    def calculate_gamma_vectorized(self, S: float, strikes: np.ndarray, T: float, r: float, sigma: float) -> np.ndarray:
        """
        Vectorized Gamma calculation for multiple strikes using NumPy operations.

        Parameters:
        S (float): Current stock price
        strikes (np.ndarray): Array of strike prices
        T (float): Time to expiration (in years)
        r (float): Risk-free rate
        sigma (float): Implied volatility

        Returns:
        np.ndarray: Array of Gamma values corresponding to each strike price
        """
        self._validate_inputs(S, strikes, T, r, sigma)
        d1_values = self._calculate_d1_vectorized(S, strikes, T, r, sigma)
        pdf_values = norm.pdf(d1_values)
        return pdf_values / (S * sigma * np.sqrt(T))

    def calculate_gamma_bulk(self, S: float, strikes: List[float], T: float, r: float, sigma: float) -> Dict[float, float]:
        """
        Calculate Gamma for multiple strikes and return as a dictionary.

        Parameters:
        S (float): Current stock price
        strikes (List[float]): List of strike prices
        T (float): Time to expiration (in years)
        r (float): Risk-free rate
        sigma (float): Implied volatility

        Returns:
        Dict[float, float]: Dictionary mapping strike prices to Gamma values
        """
        strikes_array = np.array(strikes, dtype=float)
        gamma_values = self.calculate_gamma_vectorized(S, strikes_array, T, r, sigma)
        return dict(zip(strikes, gamma_values))

# Singleton instance for convenience
gamma_calculator = BlackScholesGammaCalculator()

# Standalone functions for backward compatibility
def calculate_gamma(S: float, K: float, T: float, r: float, sigma: float) -> float:
    """Backward-compatible function for single Gamma calculation."""
    return gamma_calculator.calculate_gamma(S, K, T, r, sigma)

def calculate_gamma_bulk(S: float, strikes: List[float], T: float, r: float, sigma: float) -> Dict[float, float]:
    """Backward-compatible function for bulk Gamma calculation."""
    return gamma_calculator.calculate_gamma_bulk(S, strikes, T, r, sigma)

# Example Usage
if __name__ == "__main__":
    # Example 1: Single Gamma calculation
    try:
        gamma_value = calculate_gamma(S=150.0, K=155.0, T=45/365, r=0.05, sigma=0.30)
        print(f"Single Gamma: {gamma_value:.6f}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example 2: Vectorized Gamma calculation
    stock_price = 150.0
    strikes = np.array([145.0, 150.0, 155.0])  # ITM, ATM, OTM
    try:
        gamma_values = gamma_calculator.calculate_gamma_vectorized(
            S=stock_price, strikes=strikes, T=0.25, r=0.05, sigma=0.25
        )
        print("\nVectorized Gamma calculation:")
        for strike, gamma in zip(strikes, gamma_values):
            print(f"Strike: ${strike:.2f} | Gamma: {gamma:.6f}")
    except ValueError as e:
        print(f"Error: {e}")

    # Example 3: Bulk Gamma calculation
    try:
        bulk_results = calculate_gamma_bulk(
            S=stock_price, strikes=[145.0, 150.0, 155.0], T=0.25, r=0.05, sigma=0.25
        )
        print("\nBulk Gamma calculation:")
        for strike, gamma in bulk_results.items():
            print(f"Strike: ${strike:.2f} | Gamma: {gamma:.6f}")
    except ValueError as e:
        print(f"Error: {e}")