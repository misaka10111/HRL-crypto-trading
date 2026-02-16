Action Space: What agent can do
Observation Space: What agent can see

Exploration: Try new actions to learn about the environment
Exploitation: Use current knowledge to get the best rewards

epsilon-greedy strategy:
- With probability epsilon: choose a random action (explore)
- With probability 1-epsilon: choose the best known action (exploit)

## Features (in build_features.py)
- RSI(Relative Strength Index): a momentum oscillator that measures the speed and magnitude of recent price changes. It ranges from 0 to 100 and is primarily used to identify overbought (typically above 70) or oversold (typically below 30) conditions in the market.
- MACD(Moving Average Convergence Divergence): a trend-following momentum indicator that shows the relationship between two moving averages of an asset's price. It helps the RL agent detect changes in the strength, direction, and duration of a price trend.
- Bollinger Bands: a volatility indicator consisting of a simple moving average (the middle band) and two outer bands based on standard deviations. They help the model understand the current market volatility and determine whether prices are relatively high or low compared to historical norms.
- Logarithmic Return: calculates the continuously compounded rate of return. Unlike simple percentage returns, log returns are time-additive and statistically more stable, making them the standard input for financial machine learning models to represent price changes.