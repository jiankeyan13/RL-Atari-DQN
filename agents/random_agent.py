import random
import numpy as np


class RandomAgent:
    """
    随机策略智能体，用于基线对比
    """
    
    def __init__(self, action_dim: int):
        """
        初始化随机智能体
        
        Args:
            action_dim: 动作空间维度
        """
        self.action_dim = action_dim
        self.epsilon = 0  # 兼容其他接口
        
    def choose_action(self, state: np.ndarray) -> int:
        """
        选择随机动作
        
        Args:
            state: 当前状态
            
        Returns:
            随机选择的动作
        """
        return random.randint(0, self.action_dim - 1)
    
    def reset(self):
        """重置智能体状态（随机智能体无需重置）"""
        pass 