An option's fair value can be modeled with Black-Scholes, which prices a
European option from the underlying price, strike, time to expiry, interest
rate, and volatility. All inputs except volatility are observable, which makes
volatility the interesting one.

Implied volatility is the volatility number that makes the model price equal
the market price. Traders quote options in implied volatility precisely because
it strips out the mechanical inputs and isolates what the market believes about
future movement.

Plotting implied volatility across strikes produces the volatility smile or
skew: out-of-the-money puts usually trade at higher implied volatility than
calls, reflecting demand for crash protection. The smile's shape shifts with
market stress and is itself traded.

The option Greeks measure sensitivities: delta to the underlying price, gamma
to delta itself, vega to volatility, and theta to the passage of time. A
delta-hedged position isolates the volatility bet, earning or paying the gap
between implied and subsequently realized volatility.
