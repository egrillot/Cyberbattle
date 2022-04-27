"""This file provides the CyberBattleEnv class, heir to gym.Env, which allows you to run a simulation in the cyberbattle environment."""

from copy import deepcopy
from typing import Dict

import gym
import numpy as np
import time

from ..env.utils.network import Network
from ..env.utils.user import EnvironmentProfiles, Profile, Activity
from ..vulnerabilities.attacks import AttackSet
from .defender.tools import SiemBox


class CyberBattleEnv(gym.Env):
    """CyberBattleEnv class."""

    def __init__(
        self,
        network: Network,
        profiles: Dict[Profile, int],
        max_attack_per_outcome: int=1
    ) -> None:
        """Init.
        
        Input:
        network: the structure of the environment (Network)
        profiles: passive actors in the environment (Dict[Profile, int])
        max_attack_per_outcome: max attack remained number per outcome (int)."""
        super().__init__()

        self.__machine_list = network.get_machine_list()
        self.instance_name_to_machine = {m.get_instance_name(): m for m in self.__machine_list}
        env_profiles = EnvironmentProfiles(profiles, self.__machine_list)
        attacks = AttackSet(self.__machine_list, env_profiles, max_attack_per_outcome)
        self.__network = network
        self.__profiles = env_profiles
        self.attacks = attacks
        self.attack_count = attacks.get_attack_count()
        self.__profile_count = env_profiles.get_profile_count()
        self.services = network.get_services()

        self.id_to_attack = {a.get_id(): a for a in self.attacks.get_attacks()}

        data_sources_attacker = attacks.get_data_sources()
        data_sources_profiles = env_profiles.get_available_actions()
        total_actions = set(data_sources_attacker).union(set(data_sources_profiles))
        self.action_count = len(total_actions)

        self.id_to_actions = {i: a for i, a in enumerate(total_actions)}
        self.actions_to_id = {a: i for i, a in enumerate(total_actions)}
        self.id_to_service = {i: s for i, s in enumerate(self.services)}
        self.service_to_id = {s: i for i, s in enumerate(self.services)}

        self.SiemBox = SiemBox(env_profiles, network, self.actions_to_id, self.service_to_id)

        self.step_count = 0
    
    def reset(self) -> None:
        """Reset the environment."""
        self.__network = self.__network
        self.__profiles = self.__profiles
        self.__step_count = 0
        self.__done = False
        self.__start_time = time.time()

    def get_action_count(self) -> int:
        """Return the number of action that profiles can performed in the environment."""
        return self.action_count
    
    def get_attack_count(self) -> int:
        """Return the number of attack that the attacker can performed in the environment."""
        return self.attack_count
    
    def get_profile_count(self) -> int:
        """Return the number of profile triggering data source on the environment."""
        return self.__profile_count
    
    def step(self, display_Siem=False): #à terminer
        """Run a step time during the simulation."""
        attacker_activity = Activity(source='PC_1')
        activity_matrix = self.SiemBox.on_step(self.step_count, attacker_activity, self.__start_time, display_Siem)

        self.step_count += 1

        return activity_matrix
    
    def display_history(self, machine_instance_name: str, service: str) -> None:
        """Display the incoming traffic on the provided machine instance name by the given service."""
        self.instance_name_to_machine[machine_instance_name].display_incoming_history(service)