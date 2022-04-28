

import numpy as np

from typing import Dict, List, Tuple
from enum import Enum
from ...vulnerabilities.attacks import Attack, ActionType
from ...env.utils.network import Network
from ...env.utils.machine import Machine, get_machines_by_name
from ...vulnerabilities.outcomes import LeakedCredentials, LeakedMachineIP, Reconnaissance, Collection, Escalation, LateralMove
from ...env.utils.user import Activity


class Reward(Enum):
    """Reward class."""

    failed_attack = -10
    repeat_attack = -2
    attack_success = 5
    discovered_credential = 20
    discovered_machineIP = 5

class AttackerGoal:
    """Define the attacker goal during the simulation."""

    def __init__(
        self,
        reward: float=0,
        nb_flag: int=0
    ) -> None:
        """Init.
        
        Input:
        reward: the minimum amount of rewards the attacker must retrieve (float).
        nb_flag: the number of flag the attacker must catch (int).
        """
        self.reward = reward
        self.nb_flag = nb_flag
    
    def is_reached(self, total_rewards: float, total_flags: int) -> bool:
        """Return wheither goals are reached or not."""
        return (total_rewards >= self.reward) and (total_flags >= self.nb_flag)


class MachineActivityHistory:
    """MachineActivityHistory class to get an history of the attacker activities on the machine."""

    def __init__(self, attacks: Dict[int, List[Tuple[bool, float]]]=None, last_connection: float=None) -> None:
        """Init.
        
        attacks: dictionnary that associate for each done attack id a tuple that refers to if the attack was successfull and when did the attacker performed it (Dict[int, Tuple[bool, float]])
        last_connection: last time the attacker connected himself on the machine (float).
        """
        self.attacks = attacks
        self.last_connection = last_connection
    
    def set_last_connection(self, time_connection: float) -> None:
        """Set the last connection time."""
        self.last_connection = time_connection
    
    def update_attacks(self, attack_id: int, success: bool, attack_type: ActionType) -> bool:
        """Update the activity history and return whether the attack was already execute before or not."""
        if attack_id not in self.attacks:

            self.attacks[attack_id] = [(success, attack_type)]

            return False
        
        else:

            self.attacks[attack_id].append((success, attack_type))

            return True


class Attacker:
    """Attacker class."""

    def __init__(self,
        goals: AttackerGoal,
        attacks: Dict[int, Attack],
        services: Dict[int, str],
        network: Network,
        attacks_by_machine: Dict[str, List[Attack]]
    ) -> None:
        """Init."""
        self.goals = goals
        self.__discovered_machines: Dict[str, MachineActivityHistory] = dict()
        self.__attacks = attacks
        self.__found_credential = []
        self.__services = services
        self.__network = network
        self.__attacks_by_machine = attacks_by_machine
    
    def instance_name_by_index(self, index: int) -> str:
        """Return the machine instance name that corresponds to the index."""
        return list(self.__discovered_machines.keys())[index]
    
    def set_positions(self, instance_names: List[str]) -> None:
        """Set machines where the attacker is initially connected."""
        for name in instance_names:

            self.__discovered_machines[name] = MachineActivityHistory(
                attacks=dict(),
                last_connection=None
            )
    
    def get_attack_outcome(self, attack: Attack, machine: Machine) -> Tuple[bool, float, bool]:
        """Return if the attack is successfull, the reward executing it on the machine and if a flag has been captured."""
        instance_name = machine.get_instance_name()
        attack_id = attack.get_id()
        type_attack = attack.get_type()
        reward = 0

        if (attack_id not in [a.get_id() for a in self.__attacks_by_machine[instance_name]]) or (not machine.is_running()):

            if self.__discovered_machines[instance_name].update_attacks(attack_id, False, type_attack):

                reward += Reward.repeat_attack
            
            reward += Reward.failed_attack

            return False, reward, False

        attack_phase_name = attack.get_outcomes()[0]
        outcome = machine.get_outcome(attack_phase_name)

        if self.__discovered_machines[instance_name].update_attacks(attack_id, True, type_attack):

            reward += Reward.repeat_attack
        
        else:

            reward += outcome.get_absolute_value()

        if isinstance(outcome, LeakedCredentials):

            credentials, flag = outcome.get()

            for credential in credentials:

                if credential not in self.__found_credential:

                    cred_description = credential.get_description()
                    self.__found_credential.append(cred_description)
                    machine_instance_name = cred_description[1]

                    if machine_instance_name not in self.__discovered_machines:

                        self.__discovered_machines[machine_instance_name] = MachineActivityHistory()
                        reward += Reward.discovered_machineIP

                    reward += Reward.discovered_credential
        
        elif isinstance(outcome, LeakedMachineIP):

            discovered_machines, flag = outcome.get()

            for discovered_machine in discovered_machines:

                if discovered_machine not in self.__discovered_machines:

                    self.__discovered_machines[discovered_machine] = MachineActivityHistory()
                    reward += Reward.discovered_machineIP
        
        elif isinstance(outcome, LateralMove):

            _, flag = outcome.get()
        
        elif isinstance(outcome, Reconnaissance):

            _, flag = outcome.get()
        
        elif isinstance(outcome, Collection):

            _, flag = outcome.get()
        
        return True, reward, flag
    
    def execute_local_action(self, attacker_action: np.ndarray) -> Tuple[float, bool, int, str, str, str]:
        """Execute a local action."""
        machine_instance_name = self.instance_name_by_index[attacker_action[0]]
        machine = get_machines_by_name(machine_instance_name, self.__network.get_machine_list())[0]
        attack = self.__attacks[attacker_action[1]]

        is_successfull, reward, flag_captured = self.get_attack_outcome(attack, machine)

        if is_successfull:

            return reward

        
    def on_step(self, attacker_action: Dict[str, np.ndarray]) -> Tuple[float, Activity]:
        """Return the attacker activity."""
        if attacker_action is None:

            return Activity()
        
        if 'local' in attacker_action:

            self.execute_local_action(attacker_action['local'])
        
        elif 'remote' in attacker_action:

            self.execute_remote_action(attacker_action['remote'])
        
        else:

            self.connect(attacker_action['connect'])
        