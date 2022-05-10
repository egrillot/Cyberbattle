


import numpy as np

from typing import Dict, Tuple
from .battle_environment import CyberBattleEnv


class Agent:
    """Class that each agent trained for the attack must inherit."""

    def __init__(self, name: str, type: str) -> None:
        """Init."""
        self.name = name
        self.type = type

    def exploit(self, env: CyberBattleEnv) -> Tuple[bool, Dict[str, np.ndarray]]:
        """Define how the agent is supposed to exploit the environment to choose an action, apply its method and return the choosen action and whether it's performable in the environment or not."""
        raise NotImplementedError

    def explore(self, env: CyberBattleEnv) -> Dict[str, np.ndarray]:
        """Define how the agent will explore the environment."""
        raise NotImplementedError

    def learn(self, env: CyberBattleEnv, reward: float) -> None:
        """Define how the agent will process, after have executed its action, the environment informations to learn and optimize its further decisions."""
        raise NotImplementedError
    
    def get_descritpion(self) -> str:
        """Return the model parameters."""
        raise NotImplementedError
    
    def get_name(self) -> str:
        """Return the agent name."""
        return self.name
    
    def get_type(self) -> str:
        """Return the agent type."""
        return self.type
    
    def save(self, directory_path: str) -> None:
        """Save the agent as a pickle file with the followings structure : (params, model)."""
        raise NotImplementedError
    
    def loss(self) -> None:
        """Return the model loss."""
        raise NotImplementedError
    
    def new_episode(self) -> None:
        """Reset some parameters for the new epoch."""
        raise NotImplementedError