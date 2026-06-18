import numpy as np
from scipy.stats import norm

class BlackScholesEngine:
    """
    A vectorized Black-Scholes pricing and risk engine.
    Accepts both scalar (single number) and array inputs for high-performance calculations.
    """
    def __init__(self, S, K, T, r, sigma):
        self.S = S          # Underlying asset spot price
        self.K = K          # Option strike price
        self.T = T          # Time to maturity (in years)
        self.r = r          # Risk-free interest rate
        self.sigma = sigma  # Volatility of the underlying asset

        # Use a safe T to prevent division by zero at exact expiration
        T_safe = np.maximum(self.T, 1e-10)

        # Core probability factors
        self.d1 = (np.log(self.S / self.K) + (self.r + 0.5 * self.sigma**2) * T_safe) / (self.sigma * np.sqrt(T_safe))
        self.d2 = self.d1 - self.sigma * np.sqrt(T_safe)

    # --- PRICING METHODS ---
    
    def call_price(self):
        """Calculates the theoretical price of a European Call option."""
        return self.S * norm.cdf(self.d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)

    def put_price(self):
        """Calculates the theoretical price of a European Put option."""
        return self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2) - self.S * norm.cdf(-self.d1)

    # --- THE GREEKS (RISK SENSITIVITIES) ---
    
    def call_delta(self):
        """Rate of change of call price with respect to underlying price."""
        return norm.cdf(self.d1)

    def put_delta(self):
        """Rate of change of put price with respect to underlying price."""
        return norm.cdf(self.d1) - 1

    def gamma(self):
        """Rate of change of delta with respect to underlying price (convexity)."""
        T_safe = np.maximum(self.T, 1e-10)
        return norm.pdf(self.d1) / (self.S * self.sigma * np.sqrt(T_safe))

    def vega(self):
        """Rate of change of option price with respect to volatility."""
        T_safe = np.maximum(self.T, 1e-10)
        return self.S * norm.pdf(self.d1) * np.sqrt(T_safe)

    def call_theta(self):
        """Rate of change of call price with respect to time (time decay)."""
        T_safe = np.maximum(self.T, 1e-10)
        term1 = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(T_safe))
        term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(self.d2)
        return term1 - term2

    def put_theta(self):
        """Rate of change of put price with respect to time (time decay)."""
        T_safe = np.maximum(self.T, 1e-10)
        term1 = -(self.S * norm.pdf(self.d1) * self.sigma) / (2 * np.sqrt(T_safe))
        term2 = self.r * self.K * np.exp(-self.r * self.T) * norm.cdf(-self.d2)
        return term1 + term2

# Notice how this is now flush against the left wall, completely outside the class
if __name__ == "__main__":
    # Test with a single option: Spot=100, Strike=100, Time=1 year, Rate=5%, Vol=20%
    engine = BlackScholesEngine(S=100, K=100, T=1, r=0.05, sigma=0.20)
    
    print(f"Call Price: ${engine.call_price():.2f}")
    print(f"Put Price: ${engine.put_price():.2f}")
    print(f"Gamma: {engine.gamma():.4f}")
    print(f"Vega: {engine.vega():.4f}")
    print(f"Call Theta: {engine.call_theta():.4f}")
    print(f"Put Theta: {engine.put_theta():.4f}")
    print(f"Call Delta: {engine.call_delta():.4f}")
    print(f"Put Delta: {engine.put_delta():.4f}")