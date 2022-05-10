

import numpy as np

from typing import Dict, Tuple
from ...agent import Agent
from ...battle_environment import CyberBattleEnv


class RandomAgent(Agent):
    """RandomAgent class.
    
    This agent will only choose action randomly valid action in the environment.
    """

    def __init__(self, name: str) -> None:
        """Init."""
        type = 'random_attacker'
        super().__init__(name, type)

    def exploit(self, env: CyberBattleEnv) -> Tuple[bool, Dict[str, np.ndarray]]:
        """Choose a random valid action."""
        return True, env.sample_random_valid_action_attacker()
    
    def explore(self, env: CyberBattleEnv) -> Dict[str, np.ndarray]:
        """Choose a random valid action."""
        return env.sample_random_valid_action_attacker()
    
    def learn(self, env: CyberBattleEnv, reward: float) -> None:
        return 
    
    def get_descritpion(self) -> str:
        return 'uniform'
    
    