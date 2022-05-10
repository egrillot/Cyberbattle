"""This file provides the CyberBattleEnv class, heir to gym.Env, which allows you to run a simulation in the cyberbattle environment."""

from typing import Dict, List, Tuple

import gym
import numpy as np
import time

from gym import spaces

from ..env.utils.network import Network
from ..env.utils.sensor import SiemBox
from ..env.utils.user import EnvironmentProfiles, Profile, Activity
from ..vulnerabilities.attacks import AttackSet, Attack, ActionType
from .attacker.attacker_interface import AttackerGoal, Attacker, DiscriminateSpaces
from ..vulnerabilities.outcomes import LeakedCredentials


class AttackerBounds:
    """AttackerBounds class.
    
    This class is used to define the limits of the environment explorable by the attacker.
    The idea is to overestimate the quantities so that the model performs equally well in environments of varying sizes.
    """

    def __init__(self, maximum_machines_count: int=None, maximum_credentials_count: int=None, maximum_local_attack: int=None, maximum_remote_attack: int=None) -> None:
        """Init.
        
        Inputs:
        maximum_machines_count: the overestimated machines count, default value : None implying the value will be the number of machine in the affected environment (int)
        maximum_credentials_count: the overestimated credentials count, default value : None implying the value will be the number of credentials leakable in the affected environment (int).
        """
        self.maximum_machines_count = maximum_machines_count
        self.maximum_credentials_count = maximum_credentials_count
        self.maximum_local_attack = maximum_local_attack
        self.maximum_remote_attack = maximum_remote_attack
    
    def set_all(self, machines_count: int, credentials_count: int, local_attacks_count: int, remote_attacks_count: int) -> None:
        """Set parameters.
        
        If the corresponding parameter is None, it is set with the provided parameter.
        """
        if not self.maximum_machines_count:

            self.maximum_machines_count = machines_count
        
        if not self.maximum_credentials_count:

            self.maximum_credentials_count = credentials_count

        if not self.maximum_local_attack:

            self.maximum_local_attack = local_attacks_count
        
        if not self.maximum_remote_attack:

            self.maximum_remote_attack = remote_attacks_count


class CyberBattleEnv(gym.Env):
    """CyberBattleEnv class."""

    def __init__(
        self,
        network: Network,
        profiles: Dict[Profile, int],
        attacker_bounds = AttackerBounds(),
        max_attack_per_outcome: int=1,
        attacker_goal: Dict[str, float]={
            'reward': 10000,
            'flag': 1
        },
        winnin_reward: float=5000
    ) -> None:
        """Init.
        
        Input:
        network: the structure of the environment (Network)
        profiles: passive actors in the environment (Dict[Profile, int])
        attacker_bounds : limits of the environment explorable by the attacker (AttackerBounds)
        max_attack_per_outcome: max attack remained number per outcome (int)
        attacker_goal: it refers to the goals that the attacker must reach so the environment can be done, default value : {'reward': 0.0, 'flag': 1} (Dict[str, float])
        winnin_reward: the gifted reward for the winning agent, default value : 5000 (float).
        """
        super().__init__()

        self.__machine_list = network.get_machine_list()
        self.instance_name_to_machine = dict([(m.get_instance_name(), m) for m in self.__machine_list])
        env_profiles = EnvironmentProfiles(profiles, self.__machine_list)
        self.__network = network
        self.__name = network.name
        self.__profiles = env_profiles
        self.attacks = AttackSet(self.__machine_list, env_profiles, max_attack_per_outcome)
        self.attack_count = self.attacks.get_attack_count()
        self.__services = network.get_services()
        
        local_attack = [a for a in self.attacks.get_attacks() if a.get_type() == ActionType.LOCAL]
        remote_attack = [a for a in self.attacks.get_attacks() if a.get_type() == ActionType.REMOTE]

        self.id_to_attack: Dict[int, Attack] = {
            **dict([(i, a) for i, a in enumerate(local_attack)]),
            **dict([(i + len(local_attack), a) for i, a in enumerate(remote_attack)])
        }

        self.attacks_by_machine: Dict[str, List[Attack]] = self.attacks.get_attacks_by_machines()

        data_sources_attacker = self.attacks.get_data_sources()
        data_sources_profiles = env_profiles.get_available_actions()
        total_actions = set(data_sources_attacker).union(set(data_sources_profiles))
        self.action_count = len(total_actions)

        self.leaked_credentials = []
        
        for m in self.__machine_list:

            for outcome in m.get_outcomes():
                
                if isinstance(outcome, LeakedCredentials):

                    credentials, _ = outcome.get()

                    for cred in credentials:

                        cred_desc = cred.get_description()

                        if cred_desc not in self.leaked_credentials:

                            self.leaked_credentials.append(cred_desc)

        self.__attacker_bounds = attacker_bounds
        self.__attacker_bounds.set_all(
            machines_count=len(self.__machine_list),
            credentials_count=len(self.leaked_credentials),
            local_attacks_count=len(local_attack),
            remote_attacks_count=len(remote_attack)
        )

        self.__siembox = SiemBox(
            env_profiles,
            network,
            {a: i for i, a in enumerate(total_actions)},
            {s: i for i, s in enumerate(self.__services)},
            0
        )
        self.__attacker = Attacker(
            goals = AttackerGoal(
                reward=attacker_goal['reward'],
                nb_flag=int(attacker_goal['flag'])
            ),
            attacks=self.id_to_attack,
            network=network,
            attacks_by_machine=self.attacks_by_machine,
            start_time=0,
            credentials=[credential[2] for credential in self.leaked_credentials]
        )

        self.__winning_reward = winnin_reward

        self.attacker_action_space = DiscriminateSpaces({
            "local": spaces.MultiDiscrete(
                [self.__attacker_bounds.maximum_machines_count, self.__attacker_bounds.maximum_local_attack]),
            "remote": spaces.MultiDiscrete(
                [self.__attacker_bounds.maximum_machines_count, self.__attacker_bounds.maximum_machines_count, self.__attacker_bounds.maximum_remote_attack]),
            "connect": spaces.MultiDiscrete(
                [self.__attacker_bounds.maximum_machines_count, self.__attacker_bounds.maximum_credentials_count]),
            "submarine": spaces.MultiDiscrete([self.__attacker_bounds.maximum_machines_count])
        })

        self.reward_range = (-np.infty, np.infty)

        self.reset()
        self.check_environment()
    
    def get_name(self) -> str:
        """Return the environment name."""
        return self.__name
    
    def get_attacker_bounds(self) -> AttackerBounds:
        """Return the attacker bounds."""
        return self.__attacker_bounds
    
    def get_attacker(self) -> Attacker:
        """Return the attacker interface."""
        return self.__attacker
    
    def reset(self) -> None:
        """Reset the environment."""
        self.__network.reset()
        self.__profiles.reset()
        self.__step_count = 0
        self.__done = False
        self.__start_time = time.time()
        self.__attacker.reset(self.__start_time)
        self.__siembox.reset(self.__start_time)
        self.attacker_activity = None
        
    def display_network(self, save_figure: str=None, annotations=True) -> None:
        """Display the network structure."""
        self.__network.display(save_figure, annotations)
    
    def attacker_description(self) -> None:
        """Describe the attacker's attack wallet, its goals, discovered machines and where the attack can be performed."""
        attack_wallet = self.__attacker.attack_as_string()
        discovered_machines = self.__attacker.get_discovered_machines()
        goals = self.__attacker.goals_description()

        print(f"The attacker can performed the following attacks classified by their index :\n{attack_wallet}\n\nThese attacks can be performed and allow the attacker to get outcomes as follows :\n{self.attacks.get_attacks_by_machines_string()}\n\nThe attacker discovered and infected the following machines : {discovered_machines}.\n\n{goals}")
    
    def get_start_time(self) -> float:
        """Return the starting time of the simulation."""
        return self.__start_time
    
    def get_infected_machines(self) -> List[str]:
        """Return the current infected machines."""
        return [m.get_instance_name() for m in self.__machine_list if m.is_infected]

    def get_discovered_machines(self) -> List[str]:
        """Return the current discovered machines by the attacker."""
        return self.__attacker.get_discovered_machines()

    def get_leaked_credentials(self) -> List[Tuple[str, str, str]]:
        """Return the current leaked credentials."""
        return self.__attacker.get_discovered_credentials()
    
    def check_environment(self) -> None:
        """Check if the environment can run correctly."""
        profiles = self.__profiles.get_profiles()
        instance_names = self.instance_name_to_machine.keys()
        maximum_machines_count = self.__attacker_bounds.maximum_machines_count
        maximum_credentials_count = self.__attacker_bounds.maximum_credentials_count
        maximum_local_attacks_count = self.__attacker_bounds.maximum_local_attack
        maximum_remote_attacks_count = self.__attacker_bounds.maximum_remote_attack

        if maximum_machines_count < len(self.__machine_list):
            raise ValueError(f"The provided maximum machines count {maximum_machines_count} is lower than the total machines in the environment : {len(self.__machine_list)}.")
        
        if maximum_credentials_count < len(self.leaked_credentials):
            raise ValueError(f"The provided maximum credentials count {maximum_credentials_count} is lower than the total leakable credentials in the environment : {len(self.leaked_credentials)}.")

        local_attack_count = len([a for a in self.id_to_attack.values() if a.get_type() == ActionType.LOCAL])

        if maximum_local_attacks_count != local_attack_count:
            raise ValueError(f"The provided maximum local attacks count {maximum_local_attacks_count} isn't equal to the available local attacks in the environment : {local_attack_count}.")

        remote_attack_count = len([a for a in self.id_to_attack.values() if a.get_type() == ActionType.REMOTE])

        if maximum_remote_attacks_count != remote_attack_count:
            raise ValueError(f"The provided maximum remote attacks count {maximum_remote_attacks_count} isn't equal to the total the available remote attacks in the environment : {remote_attack_count}.")

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

    def get_step_count(self) -> int:
        """Return the current step number."""
        return self.__step_count
    
    def is_done(self) -> bool:
        """Return if the simulation is done or not."""
        return self.__done

    def attacker_step(self, attacker_action: Dict[str, np.ndarray]={'submarine': np.array([0])}) -> Tuple[bool, float]:
        """Execute the choosen action by the attacker on the environment."""
        reward, self.attacker_activity = self.__attacker.on_step(attacker_action)

        if self.__attacker.reached_goals():

            self.__done = True
            reward += self.__winning_reward

        return self.is_done(), reward
    
    def traffic_step(self, display_Siem=False) -> np.ndarray:
        """Sample an acitivity through the network and return the activity matrix."""
        if not self.attacker_activity:

            self.attacker_activity = Activity()
            
        self.__step_count += 1       

        return self.__siembox.on_step(self.__network, self.__step_count, self.attacker_activity, self.__start_time, display_Siem)

    def defender_step(self, defender_action: Dict[str, np.ndarray]) -> Tuple[bool, float]:
        """Execute the choosen action by the defender on the environment."""
        return True, 0
    
    def display_attacker_history(self, x: str='iteration') -> None:
        """Display the attacker history."""
        self.__attacker.display_history(x)
    
    def display_history(self, machine_instance_name: str, service: str) -> None:
        """Display the incoming traffic on the provided machine instance name by the given service."""
        self.instance_name_to_machine[machine_instance_name].display_incoming_history(service)

    def compare_attacker_and_traffic(self) -> None:
        """Compare the attacker attack history with the traffic"""
        attacker_history = self.__attacker.get_attacker_history()
        siem_history = self.__siembox.get_history()

    def sample_random_valid_action_attacker(self) -> Dict[str, np.ndarray]:
        """Generate a random action that can be performed in the environment."""
        return self.__attacker.sample_random_valid_action(self.attacker_action_space)

    def is_attacker_action_valid(self, attack: Dict[str, np.ndarray]) -> bool:
        """Return whether an attack is executable in the environment or not."""
        if 'submarine' in attack:

            source = attack['submarine'][0]

            return self.instance_name_to_machine[self.__attacker.instance_name_by_index(source)].is_infected

        discovered_machine_count = len(self.__attacker.get_discovered_machines())
        
        if 'local' in attack:

            if attack['local'][0] < discovered_machine_count:

                source_name = self.__attacker.instance_name_by_index(attack['local'][0])

                if self.instance_name_to_machine[source_name].is_infected:

                    if self.id_to_attack[attack['local'][1]].get_type() == ActionType.LOCAL:

                        return True
        
        if 'remote' in attack:

            target = attack['remote'][1]
            source = attack['remote'][0]
            action = attack['remote'][2]

            if self.id_to_attack[action].get_type() == ActionType.REMOTE:

                if target < discovered_machine_count:

                    if source < discovered_machine_count:

                        if self.instance_name_to_machine[self.__attacker.instance_name_by_index[source]].is_infected:

                            return True
        
        discovered_credentials_count = len(self.__attacker.get_discovered_credentials())

        if 'connect' in attack:

            if attack['connect'][0] < discovered_machine_count:

                source_name = self.__attacker.instance_name_by_index(attack['connect'][0])

                if self.instance_name_to_machine[source_name].is_infected:

                    if attack['connect'][1] < discovered_credentials_count:

                        return True

        return False