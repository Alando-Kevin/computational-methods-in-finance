import scipy.stats as si
import numpy as np
import matplotlib.pyplot as plt

# Black-Scholes formula for European call option price
def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = S * si.norm.cdf(d1) - K * np.exp(-r * T) * si.norm.cdf(d2)
    return call_price

# Vega of the call option (derivative of price with respect to volatility)
def vega(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return S * si.norm.pdf(d1) * np.sqrt(T)

# Newton-Raphson method to compute implied volatility
def implied_volatility_newton(S, K, T, r, market_price, tol=1e-8, max_iter=100):
    sigma = 0.2  # Initial guess for volatility
    for i in range(max_iter):
        price = black_scholes_call(S, K, T, r, sigma)
        vega_value = vega(S, K, T, r, sigma)
        price_diff = price - market_price
        if abs(price_diff) < tol:
            return sigma  # Convergence achieved
        sigma -= price_diff / vega_value
    raise ValueError("Newton-Raphson method did not converge within the maximum iterations")

# Parameters for the plot
S = 100  # Current stock price
T = 1    # Time to maturity (1 year)
r = 0.05  # Risk-free interest rate (5%)
market_prices = np.linspace(5, 15, 10)  # Example range of market prices for different strikes
strikes = np.arange(80, 120, 5)  # Range of strike prices

# Calculate implied volatilities for different strike prices
implied_vols = []
for K in strikes:
    # Assuming an arbitrary market price for the demonstration, which can be modified
    market_price = black_scholes_call(S, K, T, r, 0.2)  # Using 20% as a reference volatility
    try:
        iv = implied_volatility_newton(S, K, T, r, market_price)
        implied_vols.append(iv)
    except ValueError:
        implied_vols.append(np.nan)  # Append NaN if convergence fails

# Plotting the volatility smile
plt.figure(figsize=(10, 6))
plt.plot(strikes, implied_vols, marker='o', linestyle='-', color='b')
plt.xlabel('Strike Price (K)')
plt.ylabel('Implied Volatility')
plt.title('Volatility Smile')
plt.grid(True)
plt.show()
