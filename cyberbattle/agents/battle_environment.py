"""This file provides the CyberBattleEnv class, heir to gym.Env, which allows you to run a simulation in the cyberbattle environment."""

from typing import Dict, List

import gym
import numpy as np
import time

from ..env.utils.network import Network
from ..env.utils.sensor import SiemBox
from ..env.utils.user import EnvironmentProfiles, Profile
from ..vulnerabilities.attacks import AttackSet, Attack

from .attacker.attacker_interface import AttackerGoal, Attacker


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
        self.__services = network.get_services()

        self.id_to_attack: Dict[int, Attack] = {a.get_id(): a for a in self.attacks.get_attacks()}
        self.attacks_by_machine: Dict[str, List[Attack]] = attacks.get_attacks_by_machines()

        data_sources_attacker = attacks.get_data_sources()
        data_sources_profiles = env_profiles.get_available_actions()
        total_actions = set(data_sources_attacker).union(set(data_sources_profiles))
        self.action_count = len(total_actions)

        self.id_to_actions = {i: a for i, a in enumerate(total_actions)}
        self.actions_to_id = {a: i for i, a in enumerate(total_actions)}
        self.id_to_service = {i: s for i, s in enumerate(self.__services)}
        self.service_to_id = {s: i for i, s in enumerate(self.__services)}

        self.__siembox = SiemBox(env_profiles, network, self.actions_to_id, self.service_to_id, 0)
        self.__attacker = Attacker(
            goals = AttackerGoal(
                reward=0,
                nb_flag=1
            ),
            attacks=self.id_to_attack,
            network=network,
            attacks_by_machine=self.attacks_by_machine,
            start_time=0
        )

        self.reset()
    
    def reset(self) -> None:
        """Reset the environment."""
        self.__network.reset()
        self.__profiles.reset()
        self.__step_count = 0
        self.__done = False
        self.__start_time = time.time()
        self.__attacker.reset(self.__start_time)
        self.__siembox.reset(self.__start_time)
    
    def get_start_time(self) -> float:
        """Return the starting time of the simulation."""
        return self.__start_time
    
    def check_environment(self) -> None:
        """Check if the environment can run correctly."""
        profiles = self.__profiles.get_profiles().keys()
        instance_names = self.instance_name_to_machine.keys()

        for profile in profiles:

            for instance_name in profile.preferences.source_prior.keys():

                if instance_name not in instance_names:
                    raise ValueError(f"The user {profile.get_instance_name()} can perform actions on machine {instance_name} but the network doesn't have any machines with this instance name.")

            for instance_name in profile.preferences.target_prior.keys():

                if instance_name not in instance_names:
                    raise ValueError(f"The user {profile.get_instance_name()} can perform actions on machine {instance_name} but the network doesn't have any machines with this instance name.")

    def get_action_count(self) -> int:
        """Return the number of action that profiles can performed in the environment."""
        return self.action_count
    
    def get_attack_count(self) -> int:
        """Return the number of attack that the attacker can performed in the environment."""
        return self.attack_count
    
    def get_profile_count(self) -> int:
        """Return the number of profile triggering data source on the environment."""
        return self.__profile_count

    def get_step_count(self) -> int:
        """Return the current step number."""
        return self.__step_count
    
    def is_done(self) -> bool:
        """Return if the simulation is done or not."""
        return self.__done

    def attacker_step(self, attacker_action: Dict[str, np.ndarray], display_Siem=False): #à terminer
        """Run a step time during the simulation."""
        reward, attacker_activity = self.__attacker.on_step(attacker_action)
        activity_matrix = self.__siembox.on_step(self.__network, self.__step_count, attacker_activity, self.__start_time, display_Siem)

        if self.__attacker.reached_goals():

            self.__done = True

        self.__step_count += 1
    
    def display_history(self, machine_instance_name: str, service: str) -> None:
        """Display the incoming traffic on the provided machine instance name by the given service."""
        self.instance_name_to_machine[machine_instance_name].display_incoming_history(service)

    def compare_attacker_and_traffic(self) -> None:
        """Compare the attacker attack history with the traffic"""
        attacker_history = self.__attacker.get_attacker_history()
        siem_history = self.__siembox.get_history()
