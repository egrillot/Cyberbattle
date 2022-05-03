


import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import pandas as pd
import random
import time

from typing import Dict, List, Tuple
from ...vulnerabilities.attacks import Attack, ActionType
from ...env.utils.network import Network
from ...env.utils.machine import Machine, get_machines_by_name, trouble
from ...vulnerabilities.outcomes import LeakedCredentials, LeakedMachineIP, Reconnaissance, Collection, Escalation, LateralMove
from ...env.utils.user import Activity
from ...env.utils.flow import Error, Credential


class Reward:
    """Reward class."""

    failed_attack = -10
    repeat_attack = -2
    succeeded_attack = 5
    discovered_credential = 20
    discovered_machineIP = 5
    connection = 50
    flag_captured = 50
    escalation = 10

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

    def __init__(self, last_connection: float=None, attacks: Dict[int, List[Tuple[bool, float]]]=dict()) -> None:
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
        network: Network,
        attacks_by_machine: Dict[str, List[Attack]],
        start_time: float
    ) -> None:
        """Init.
        
        Input:
        attacks: dictionnary that links a choosen index by the attacker agent to an attack (Dict[int, Attack]),
        network: the network structure (Network),
        attacks_by_machine: dictionnary that links a machine instance name to the executable attacks on it (Dict[str, List[Attack]]),
        start_time: (float).
        """
        self.goals = goals
        self.__discovered_machines: Dict[str, MachineActivityHistory] = dict()
        self.__attacks = attacks
        self.__found_credential: List[Credential] = []
        self.__network = network
        self.__attacks_by_machine = attacks_by_machine
        self.__start_time = start_time
        self.__captured_flag = 0
        self.__reward = 0
        self.__cumulative_reward = []
        self.history: Dict[str, List[object]] = {
            'time': [],
            'attack name': [],
            'machine instance name': [],
            'reward': [],
            'result': [],
            'type': [],
            'flag': []
        }

        self.set_positions([m.get_instance_name() for m in network.get_machine_list() if m.is_infected])
    
    def instance_name_by_index(self, index: int) -> str:
        """Return the machine instance name that corresponds to the index."""
        return list(self.__discovered_machines.keys())[index]

    def get_discovered_machines(self) -> List[str]:
        """Return the discovered machines and their associated activity history."""
        return list(self.__discovered_machines.keys())
    
    def attack_as_string(self) -> Dict[int, str]:
        """Return the attacks as string by index."""
        return dict([(i, a.get_name()) for i, a in self.__attacks.items()])
    
    def get_cumulative_rewards(self) -> np.ndarray:
        """Return the cumulative reward."""
        return np.cumsum(np.array(self.__cumulative_reward))
    
    def get_total_reward(self) -> float:
        """Return the cumulative reward."""
        return self.__reward
    
    def set_positions(self, instance_names: List[str]) -> None:
        """Set machines where the attacker is initially connected."""
        for name in instance_names:

            self.__discovered_machines[name] = MachineActivityHistory(
                last_connection=self.__start_time
            )
        
    def is_discovered(self, machine: str):
        """Return whether a machine has been discovered or not."""
        return machine in self.__discovered_machines

    def get_discovered_credentials(self) -> List[Tuple[str, str, str]]:
        """Return the discovered credential."""
        return [cred.get_description() for cred in self.__found_credential]
    
    def update_history(self, time: float, reward: float, attack_name: str, machine_instance_name: str, result: str, type: str, flag: bool) -> None:
        """Update the attack history."""
        self.history['time'].append(time - self.__start_time)
        self.history['reward'].append(reward)
        self.history['attack name'].append(attack_name)
        self.history['machine instance name'].append(machine_instance_name)
        self.history['result'].append(result)
        self.history['type'].append(type)
        self.history['flag'].append(flag)

    def get_infected_machines(self) -> List[Tuple[str, float]]:
        """Return the infected machine instance name with the last connection time."""
        return [(instance_name, self.__discovered_machines[instance_name].last_connection - self.__start_time) for instance_name in self.__discovered_machines if self.__discovered_machines[instance_name].last_connection]
    
    def get_attack_outcome(self, attack: Attack, machine: Machine) -> Tuple[bool, float, bool]:
        """Return whether the attack has been performed successfully, the reward executing it on the machine and whether a flag has been captured or not."""
        instance_name = machine.get_instance_name()
        attack_id = attack.get_id()
        type_attack = attack.get_type()
        reward = 0
        
        if (attack_id not in [a.get_id() for a in self.__attacks_by_machine[instance_name]]) or (not machine.is_running()):

            if self.__discovered_machines[instance_name].update_attacks(attack_id, False, type_attack):

                reward += Reward.repeat_attack
            
            reward += Reward.failed_attack

            return False, reward, False

        attack_phase_names = attack.get_outcomes()

        for phase_name in attack_phase_names:

            outcomes = machine.get_outcome(phase_name)

            already_done = self.__discovered_machines[instance_name].update_attacks(attack_id, True, type_attack)

            if already_done:
                
                reward += Reward.repeat_attack
            
            for outcome in outcomes:

                if not already_done:

                    reward += outcome.get_absolute_value()

                if isinstance(outcome, LeakedCredentials):

                    credentials, flag = outcome.get()

                    for credential in credentials:

                        if credential not in self.__found_credential:

                            self.__found_credential.append(credential)
                            machine_instance_name = credential.machine

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
                
                elif isinstance(outcome, Escalation):

                    user_right, flag = outcome.get()

                    if machine.get_attacker_right().get_right().value < user_right.value:

                        reward += Reward.escalation

                    machine.update_attacker_right(user_right)
        
        return True, reward, flag
    
    def execute_local_action(self, attacker_action: np.ndarray) -> Tuple[float, bool, Activity]:
        """Execute a local action."""
        machine_instance_name = self.instance_name_by_index(attacker_action[0])        
        attack = self.__attacks[attacker_action[1]]

        if machine_instance_name not in self.__discovered_machines:
            raise ValueError(f"The machine {machine_instance_name} isn't discovered yet.")
        
        if attack.get_type() == ActionType.REMOTE:
            raise ValueError("To execute a local action, you need to used an attack with a local type.")
        
        if not self.__discovered_machines[machine_instance_name].last_connection:

            reward = Reward.failed_attack
            self.update_history(time.time(), reward, attack.get_name(), machine_instance_name, 'failed', 'local', False)

            return reward, False, Activity()

        machine = get_machines_by_name(machine_instance_name, self.__network.get_machine_list())[0]
        
        is_successfull, reward, flag = self.get_attack_outcome(attack, machine)

        if is_successfull:

            port, action = random.choice([(machine.get_service_name(ds), ds) for ds in attack.get_data_sources() if machine.is_data_source_available(ds)])

            activity = Activity(
                source=machine_instance_name,
                activity=True,
                action=action,
                where=machine_instance_name,
                service=port,
                error=Error.NO_ERROR
            )
            
            self.update_history(time.time(), reward, attack.get_name(), machine.get_instance_name(), 'successfull', 'local', flag)

            return reward, flag, activity
        
        self.update_history(time.time(), reward, attack.get_name(), machine.get_instance_name(), 'failed', 'local', flag)
        
        return reward, flag, Activity()

    def execute_remote_action(self, attacker_action: np.ndarray) -> Tuple[float, bool, Activity]:
        """Execute a remote action."""
        source = self.instance_name_by_index(attacker_action[0])
        target = self.instance_name_by_index(attacker_action[1])        
        
        if source == target:
            raise ValueError(f"To execute a remote action, the source and the target must be different, the source and target provided are respectively {source} and {target}.")
        
        if source not in self.__discovered_machines:
            raise ValueError(f"The machine {source} isn't discovered yet.")

        if target not in self.__discovered_machines:
            raise ValueError(f"The machine {target} isn't discovered yet.")

        if not self.__discovered_machines[source].last_connection:

            return Reward.failed_attack, False, Activity()

        attack = self.__attacks[attacker_action[2]]

        if attack.get_type() == ActionType.LOCAL:
            raise ValueError("To execute a remote action, you need to used an attack with a remote type.")

        target_machine = get_machines_by_name(target, self.__network.get_machine_list())[0]

        path = self.__network.get_path(source, target)

        ports_and_actions = [(target_machine.get_service_name(ds), ds) for ds in attack.get_data_sources() if target_machine.is_data_source_available(ds)]

        if len(ports_and_actions) == 0:

            reward = Reward.failed_attack
            self.update_history(time.time(), reward, attack.get_name(), target, 'failed', 'remote', False)

            return reward, False, Activity()

        else:

            port, action = random.choice(ports_and_actions)
            error = Error.NO_ERROR if isinstance(path, int) else trouble(path, port)

            if error.value > 0:

                activity = Activity(
                    source=source,
                    activity=True,
                    action=action,
                    where=target,
                    service=port,
                    error=error
                )

                self.update_history(time.time(), 0, attack.get_name(), target, 'network failed', 'remote', False)

                return 0, False, activity

            is_successfull, reward, flag = self.get_attack_outcome(attack, target_machine)

            if is_successfull:

                activity = Activity(
                    source=source,
                    activity=True,
                    action=action,
                    where=target,
                    service=port,
                    error=Error.NO_ERROR
                )

                self.update_history(time.time(), reward, attack.get_name(), target, 'successfull', 'remote', flag)

                return reward, flag, activity

            self.update_history(time.time(), reward, attack.get_name(), target, 'failed', 'remote', flag)

            return reward, flag, Activity()
    
    def connect(self, attacker_action: np.ndarray) -> Tuple[float, bool, Activity]:
        """Connect."""
        source = self.instance_name_by_index(attacker_action[0])

        if source not in self.__discovered_machines:
            raise ValueError(f"The machine {source} isn't discovered yet.")

        if not self.__discovered_machines[source].last_connection:

            reward = Reward.failed_attack
            self.update_history(time.time(), reward, 'User Account Authentification', target, 'failed', 'connect', False)

            return reward, False, Activity()
        
        credential = self.__found_credential[attacker_action[1]]
        target = credential.machine

        if target not in self.__discovered_machines:
            raise ValueError(f"The machine {target} isn't discovered yet.")
        
        service = credential.port
        path = self.__network.get_path(source, target)

        if isinstance(path, int):

            if self.__discovered_machines[target].last_connection:

                self.__discovered_machines[target].last_connection = time.time()
                reward = Reward.repeat_attack
                self.update_history(time.time(), reward, 'User Account Authentification', target, 'repeat', 'connect', False)

                return reward, False, Activity() 
        
        error = trouble(path, service)

        if error.value > 0:

            activity = Activity(
                source=source,
                activity=True,
                where=target,
                action='User Account: User Account Authentification',
                service=service,
                error=error
            )

            self.update_history(time.time(), 0, 'User Account Authentification', target, 'network failed', 'connect', False)

            return 0, False, activity

        reward = 0

        if not self.__discovered_machines[target].last_connection:
            
            self.__discovered_machines[target].last_connection = time.time()
            reward += Reward.connection
            target_machine = path[-1]
            reward += target_machine.get_value()
            target_machine.infect()
            flag = target_machine.is_flag()
        
        activity = Activity(
                source=source,
                activity=True,
                where=target,
                action='User Account: User Account Authentification',
                service=service,
                error=error
        )

        self.update_history(time.time(), reward, 'User Account Authentification', target, 'successfull', 'connect', flag)
        
        return reward, flag, activity
        
    def on_step(self, attacker_action: Dict[str, np.ndarray]) -> Tuple[float, Activity]:
        """Return the attacker activity."""
        if 'connect' in attacker_action:

            reward, flag, activity = self.connect(attacker_action['connect'])

        elif 'submarine' in attacker_action:

            return 0, Activity()
        
        elif 'local' in attacker_action:

            reward, flag, activity = self.execute_local_action(attacker_action['local'])
        
        elif 'remote' in attacker_action:

            reward, flag, activity = self.execute_remote_action(attacker_action['remote'])
        
        if flag:

            self.__captured_flag += 1
        
        self.__reward += reward
        self.__cumulative_reward.append(reward)
        
        return reward, activity
    
    def get_captured_flag(self) -> int:
        """Return the current number of captured flags."""
        return self.__captured_flag
    
    def reached_goals(self) -> bool:
        """Return whether the goals are reached or not."""
        return self.goals.is_reached(self.__reward, self.__captured_flag)
        
    def reset(self, new_start_time: float) -> None:
        """Reset the attacker attributes."""

        self.__found_credential: List[Credential] = []
        self.__discovered_machines: Dict[str, MachineActivityHistory] = dict()
        self.__network.reset()
        self.__start_time = new_start_time
        self.__captured_flag = 0
        self.__reward = 0
        self.__cumulative_reward = []

        self.set_positions([m.get_instance_name() for m in self.__network.get_machine_list() if m.is_infected])
    
    def display_history(self) -> None:
        """Display the attacker action history through a time line."""
        self.history['cumulative reward'] = self.get_cumulative_rewards()
        df = pd.DataFrame(self.history, columns=['time', 'reward', 'attack name', 'machine instance name', 'result', 'type', 'flag', 'cumulative reward'])
        fig = px.scatter(df, x='time', y='cumulative reward', color='result', symbol='type', title="Attacker history", hover_data=['reward', 'attack name', 'machine instance name', 'type', 'flag'])
        #fig.update_traces(mode="markers+lines")
        fig.show()
    
    def get_attacker_history(self) -> Dict[str, object]:
        """Return the attacker attack history."""
        self.history['cumulative reward'] = self.get_cumulative_rewards()

        return self.history