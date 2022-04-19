"""This file provides the CyberBattleEnv class, heir to gym.Env, which allows you to run a simulation in the cyberbattle environment."""

import gym
import numpy as np

from ..env.utils.network import Network
from ..env.utils.user import EnvironmentProfiles
from ..vulnerabilities.attacks import AttackSet

class CyberBattleEnv(gym.Env):
    """CyberBattleEnv class."""

    def __init__(
        self,
        network: Network,
        profiles: EnvironmentProfiles,
        attacks: AttackSet 
    ) -> None:
        """Init.
        
        Input:
        network: the structure of the environment (Network)
        profiles: passive actors in the environment (EnvironmentProfiles)
        attacks: the attack the attacker can performed in the environment (Attackset)."""
        super().__init__()

        self.network = network
        self.profiles = profiles
        self.attacks = attacks
        self.attack_count = attacks.get_attack_count()
        self.profile_count = profiles.get_profile_count()

        self.id_to_attack = {i: a for (i, a) in zip(range(self.attack_count), self.attacks.get_attacks())}
        self.attack_to_id = {a: i for (i, a) in zip(range(self.attack_count), self.attacks.get_attacks())}

        data_sources_attacker = attacks.get_data_sources()
        data_sources_profiles = profiles.get_available_actions()
        total_actions = set(data_sources_attacker).union(set(data_sources_profiles))
        self.action_count = len(total_actions)

        self.id_to_actions = {i: a for i, a in enumerate(total_actions)}
        self.actions_to_id = {a: i for i, a in enumerate(total_actions)}

    def get_action_count(self) -> int:
        """Return the number of action that profiles can performed in the environment."""
        return self.action_count
    
    def get_attack_count(self) -> int:
        """Return the number of attack that the attacker can performed in the environment."""
        return self.attack_count