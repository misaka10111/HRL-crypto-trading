import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import math
from stable_baselines3 import SAC
from sac_goal import GoalConditionedCryptoEnv


class HighLevelCryptoEnv(gym.Env):
    """
    Step 4: HIRO-like 高层管理器环境。
    它输出目标仓位 (Goal)，并调用冻结的底层 SAC 模型去执行 c 步。
    """
    metadata = {'render_modes': ['human', 'console']}

    def __init__(self, df: pd.DataFrame, low_level_model_path: str, macro_step_freq=48, initial_balance=10000.0):
        super(HighLevelCryptoEnv, self).__init__()
        
        self.df = df
        self.macro_step_freq = macro_step_freq
        self.initial_balance = initial_balance
        self.num_crypto_assets = 1
        self.total_assets = self.num_crypto_assets + 1 
        
        self.low_level_env = GoalConditionedCryptoEnv(
            df=self.df, 
            initial_balance=self.initial_balance, 
            goal_change_freq=self.macro_step_freq
        )
        
        print(f"Loading pre-trained low-level worker from {low_level_model_path}...")
        self.low_level_model = SAC.load(low_level_model_path)
        
        # 3. 高层动作空间：输出给底层的 Goal (目标仓位权重)
        self.action_space = spaces.Box(
            low=0.0, high=1.0, 
            shape=(self.total_assets,), 
            dtype=np.float32
        )
        
        # 4. 高层状态空间：(这里为了简化，只看当前市场特征和真实仓位)
        self.num_market_features = len(self.df.columns)
        total_obs_dim = self.num_market_features + self.total_assets
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, 
            shape=(total_obs_dim,), 
            dtype=np.float32
        )

        # 5. 高层风控记录 (融合了 Step 2 修复后的逻辑)
        self.peak_portfolio_value = self.initial_balance
        self.previous_drawdown = 0.0
        self.risk_penalty_weight = 0.5
        
        self.current_low_obs = None

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 重置底层环境，获取底层的初始 observation
        self.current_low_obs, info = self.low_level_env.reset(seed=seed)
        
        # 重置高层风控状态
        self.peak_portfolio_value = self.initial_balance
        self.previous_drawdown = 0.0
        
        return self._get_high_level_obs(), info

    def step(self, macro_action: np.ndarray):
        """
        核心逻辑：高层下达 Goal, 底层循环执行 macro_step_freq 次
        """
        # 1. 规范化高层的动作，将其作为底层的 Goal
        goal_weights = np.clip(macro_action, 0.0, 1.0)
        weight_sum = np.sum(goal_weights)
        if weight_sum > 0:
            goal_weights = goal_weights / weight_sum
        else:
            goal_weights = np.zeros(self.total_assets)
            goal_weights[0] = 1.0
            
        # 强行将这个 Goal 注入到底层环境中
        self.low_level_env.current_goal_weights = goal_weights
        
        # 记录高层决策起点的资产，用于计算 48 步之后的宏观收益
        portfolio_value_start = self.low_level_env.portfolio_value
        
        terminated = False
        truncated = False
        macro_steps_taken = 0
        
        # ==========================================
        # 2. 底层 Worker 执行循环 (The Inner Loop)
        # ==========================================
        for _ in range(self.macro_step_freq):
            # 将最新的 Goal 更新到底层的 observation 中
            self.current_low_obs[-self.total_assets:] = goal_weights # type: ignore
            
            # 询问冻结的底层 SAC 模型：为了达到这个 Goal，我现在该怎么买卖？
            # 注意：deterministic=True 让底层纯粹执行，不再做随机探索
            low_level_action, _ = self.low_level_model.predict(self.current_low_obs, deterministic=True) # type: ignore
            
            # 在底层物理环境中执行
            self.current_low_obs, low_reward, terminated, truncated, info = self.low_level_env.step(low_level_action)
            
            macro_steps_taken += 1
            if terminated or truncated:
                break
                
        # ==========================================
        # 3. 计算高层管理者的 Reward (结合 Step 2 风控)
        # ==========================================
        portfolio_value_end = self.low_level_env.portfolio_value
        
        if portfolio_value_start > 0:
            # A. 宏观基础收益 (这 c 步内的总对数收益)
            macro_log_return = math.log(portfolio_value_end / portfolio_value_start)
            macro_base_reward = macro_log_return * 100.0
            
            # B. 宏观风险惩罚 (Step 2 修复版的增量回撤惩罚)
            if portfolio_value_end > self.peak_portfolio_value:
                self.peak_portfolio_value = portfolio_value_end
                
            current_drawdown = (self.peak_portfolio_value - portfolio_value_end) / self.peak_portfolio_value
            drawdown_delta = current_drawdown - self.previous_drawdown
            
            if drawdown_delta > 0:
                macro_drawdown_penalty = self.risk_penalty_weight * drawdown_delta * 100.0
            else:
                macro_drawdown_penalty = 0.0
                
            self.previous_drawdown = current_drawdown
            
            # 最终高层 Reward：在这 4 小时里，你选的方向让我赚钱了吗？风控做好了吗？
            macro_reward = macro_base_reward - macro_drawdown_penalty
        else:
            macro_reward = -1.0
            
        return self._get_high_level_obs(), macro_reward, terminated, truncated, info

    def _get_high_level_obs(self):
        # 从底层环境中抽取出特征和真实的当前仓位，供高层做下一次决策
        # 前 num_market_features 个是市场特征
        features = self.current_low_obs[:self.num_market_features] # type: ignore
        # 中间的 total_assets 是实际仓位
        actual_weights = self.current_low_obs[self.num_market_features : self.num_market_features + self.total_assets] # type: ignore
        
        return np.concatenate([features, actual_weights])
    

if __name__ == "__main__":
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    from stable_baselines3.common.callbacks import BaseCallback
    import os

    # 1. 加载数据
    print("正在加载真实历史行情数据...")
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(BASE_DIR, 'btcusd_5-min_data.csv')
    df = pd.read_csv(data_path, index_col="Datetime", parse_dates=True).sort_index()

    # 划分训练集和测试集
    split_idx = int(len(df) * 0.8)
    train_df = df.iloc[:split_idx].reset_index(drop=True)

    # 2. 指定底层模型的路径 (必须先运行 Step 3 生成此文件)
    # 假设你在 Step 3 保存的模型名为 sac_low_level_executioner.zip
    LOW_LEVEL_MODEL_PATH = os.path.join(BASE_DIR, "sac_low_level_executioner.zip")
    
    if not os.path.exists(LOW_LEVEL_MODEL_PATH):
        raise FileNotFoundError(f"找不到底层模型文件：{LOW_LEVEL_MODEL_PATH}。请先完成 Step 3 的训练。")

    # 3. 实例化高层环境
    # macro_step_freq=48 表示高层每 48 个底层步 (例如 48*5=240分钟=4小时) 决策一次
    base_env = HighLevelCryptoEnv(
        df=train_df, 
        low_level_model_path=LOW_LEVEL_MODEL_PATH, 
        macro_step_freq=48, 
        initial_balance=10000.0
    )
    
    # 包装环境用于监控和向量化
    base_env = Monitor(base_env)
    vec_env = DummyVecEnv([lambda: base_env])
    
    # 【注意】因为高层的 Observation 包含了 Goal 和实际权重（不能被破坏分布），
    # 且我们在环境中应该处理好特征的标准化，所以这里 norm_obs=False。
    # 我们只对高层的 Reward 进行标准化，稳定训练。
    env = VecNormalize(vec_env, norm_obs=False, norm_reward=True, clip_reward=10.0)

    # 4. 初始化高层 SAC Agent
    # 高层的学习率通常可以设置得稍微小一点，因为它观察到的样本数比底层少 (总步数 / 48)
    model = SAC(
        "MlpPolicy", 
        env, 
        learning_rate=1e-4, 
        batch_size=256,
        verbose=1, 
        tensorboard_log="./sac_hiro_manager_tensorboard/"
    )
    
    print("Training High-level Manager...")
    # 因为高层每步等价于底层走 48 步，所以 total_timesteps 应该相应减少
    # len(train_df) // 48 就是高层在一个 epoch 里能走的步数
    high_level_steps_per_epoch = len(train_df) // 48
    
    model.learn(
        total_timesteps=high_level_steps_per_epoch * 5, # 跑 5 个 epoch
        log_interval=1
    )
    
    # 6. 保存高层模型和环境状态
    print("训练完成！正在保存高层模型...")
    model.save("sac_high_level_manager")
    env.save("vec_normalize_hiro.pkl")