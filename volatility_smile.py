import scipy.stats as si
import numpy as np

# Black-Scholes formula for European call option price
def black_scholes_call(S, K, T, r, sigma):
    """
    S : float : Current stock price
    K : float : Strike price
    T : float : Time to maturity
    r : float : Risk-free interest rate
    sigma : float : Volatility of the underlying stock
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    return call_price

# Vega of the call option (derivative of price with respect to volatility)
def vega(S, K, T, r, sigma):
    """
    S : float : Current stock price
    K : float : Strike price
    T : float : Time to maturity
    r : float : Risk-free interest rate
    sigma : float : Volatility of the underlying stock
    """
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * si.norm.pdf(d1) * np.sqrt(T)

# Newton-Raphson method to compute implied volatility
def implied_volatility_newton(S, K, T, r, market_price, tol=1e-8, max_iter=100):
    """
    S : float : Current stock price
    K : float : Strike price
    T : float : Time to maturity
    r : float : Risk-free interest rate
    market_price : float : Observed market price of the call option
    tol : float : Tolerance for convergence
    max_iter : int : Maximum number of iterations
    """
    # Initial guess for volatility
    sigma = 0.2
    for i in range(max_iter):
        # Calculate the option price and vega for the current guess of sigma
        price = black_scholes_call(S, K, T, r, sigma)
        vega_value = vega(S, K, T, r, sigma)

        # Update sigma using Newton-Raphson method
        price_diff = price - market_price
        if abs(price_diff) < tol:
            return sigma  # Convergence achieved
        
        sigma -= price_diff / vega_value

    # If max iterations are reached without convergence
    raise ValueError("Newton-Raphson method did not converge within the maximum iterations")

# Example usage
S = 100  # Current stock price
K = 100  # Strike price
T = 1    # Time to maturity (1 year)
r = 0.05  # Risk-free interest rate (5%)
market_price = 10  # Observed market price of the call option

# Compute implied volatility
try:
    implied_vol = implied_volatility_newton(S, K, T, r, market_price)
    print(f"Implied Volatility: {implied_vol:.4f}")
except ValueError as e:
    print(str(e))
