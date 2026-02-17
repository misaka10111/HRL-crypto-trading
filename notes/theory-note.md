# Theory

## General

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

## Algorithms

- [HIRO](https://arxiv.org/abs/1805.08296): Hierarchical Reinforcement Learning with Off-policy Correction is a two-level hierarchical RL algorithm where a high-level policy sets goals and a low-level policy learns to achieve them, trained in an off-policy manner.

    ```text
    High-level policy (Manager)
            ↓  (goal g)
    Low-level policy (Worker)
            ↓  (action a)
    Environment
    ```

- [PPO](https://arxiv.org/abs/1707.06347): Proximal Policy Optimization algorithms are a family of policy gradient methods that use a surrogate objective function to enable stable and efficient policy updates by keeping the new policy close to the old policy.
- [SAC](https://arxiv.org/abs/1812.05905v2): Soft Actor-Critic (SAC) is an off-policy actor-critic reinforcement learning algorithm based on the maximum entropy framework, which optimizes a stochastic policy to maximize both expected return and policy entropy.
- PPO limit strategy variation; SAC encourage strategy to remain random

- on-policy: data from current policy
- off-policy: data from old/random/others' policy

- MDP: Markov Decision Process, used to describe "how to make continuous decisions in an uncertain environment"
  - (S,A,P,R,γ): state, action, transition, reward, discount factor

- batch: the amount of data used in one training step (e.g., 32 samples)  
- iteration: one iteration is counted each time a batch is trained  
- epoch: the number of times the entire training dataset (all batches) is passed through once
