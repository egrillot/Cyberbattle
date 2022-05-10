


import numpy as np
import plotly.express as px
import pandas as pd
import random
import time


from typing import Dict, List, Tuple
from gym import spaces

from ...vulnerabilities.attacks import Attack, ActionType
from ...env.utils.network import Network
from ...env.utils.machine import Machine, get_machines_by_name, trouble
from ...vulnerabilities.outcomes import LeakedCredentials, LeakedMachineIP, Reconnaissance, Collection, Escalation, LateralMove
from ...env.utils.user import Activity
from ...env.utils.flow import Error, Credential


class DiscriminateSpaces(spaces.Dict):
    """DiscriminateSpaces class.
    
    This class allow us to create an action space for the attacker that will be use to sample a random valid action. 
    Instead of the clasic spaces.Dict sampling which sample over each space, we sample only on one space.
    The aim here is also to avoid to genereate uniformly an action type but instead of that, generate an action type with respect to the spaces dimension.
    """

    def __init__(self, spaces: Dict[str, spaces.MultiDiscrete], **spaces_kwargs) -> None:
        """Init."""
        self.spaces = spaces
        self.dims: Dict[str, float] = dict([(name, np.prod(space.nvec)) for name, space in self.spaces.items()])
        super().__init__(spaces, **spaces_kwargs)

    def nvec(self) -> Dict[str, float]:
        """Return the dimension for each space in the dictionnary."""
        return self.dims
    
    def sample_action_type(self, range_action_type: List[str]) -> str:
        """Sample an action type from the provided action types with respect to the spaces proportion."""
        if not set(range_action_type).issubset(set(self.spaces.keys())):
            raise ValueError(f"Please provide action types from the following list : {list(self.spaces.keys())}. Provided action types : {range_action_type}.")
        
        new_spaces_dims_dict = np.zeros((len(range_action_type), ))

        for i, action_type in enumerate(range_action_type):
            
            new_spaces_dims_dict[i] = self.dims[action_type]
        
        new_spaces_dims_dict /= np.sum(new_spaces_dims_dict)

        return np.random.choice(range_action_type, p=new_spaces_dims_dict)


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
    waiting = -1

class AttackerGoal:
    """Define the attacker goal during the simulation."""

    def __init__(
        self,
        reward: float=10000,
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
        return (total_rewards >= self.reward) or (total_flags >= self.nb_flag)


class Attacker:
    """Attacker class."""

    def __init__(self,
        goals: AttackerGoal,
        attacks: Dict[int, Attack],
        network: Network,
        attacks_by_machine: Dict[str, List[Attack]],
        start_time: float,
        credentials: List[str]
    ) -> None:
        """Init.
        
        Input:
        attacks: dictionnary that links a choosen index by the attacker agent to an attack (Dict[int, Attack]),
        network: the network structure (Network),
        attacks_by_machine: dictionnary that links a machine instance name to the executable attacks on it (Dict[str, List[Attack]]),
        start_time: the simulation starting time (float)
        credentials: List of available credentials in the environment (List[str]).
        """
        self.goals = goals
        self.__discovered_machines: List[str] = []
        self.__last_connections: Dict[str, float] = dict()
        self.__attacks_tried: Dict[str, Dict[int, bool]] = dict()
        self.__attacks = attacks
        self.__local_attack_index = [i for i, a in attacks.items() if a.get_type() == ActionType.LOCAL]
        self.__remote_attack_index = [i for i, a in attacks.items() if a.get_type() == ActionType.REMOTE]
        self.__attacks_name_to_id = dict([(a.get_name(), i) for i, a in attacks.items()])
        nb_attack = len(self.__attacks_name_to_id)

        for i, cred in enumerate(credentials):

            self.__attacks_name_to_id[cred] = nb_attack + i

        self.__found_properties: List[str]= []
        self.__found_credential: List[Credential] = []
        self.__network = network
        self.__attacks_by_machine = attacks_by_machine
        
        self.__start_time = start_time
        self.__captured_flag = 0
        self.__reward = 0
        self.step_count = 1
        self.__cumulative_reward = []
        self.history: Dict[str, List[object]] = {
            'time': [],
            'iteration': [],
            'attack name': [],
            'machine instance name': [],
            'reward': [],
            'result': [],
            'type': [],
            'flag': [],
            'cumulative reward': []
        }

        self.set_positions([m.get_instance_name() for m in network.get_machine_list() if m.is_infected])
    
    def instance_name_by_index(self, index: int) -> str:
        """Return the machine instance name that corresponds to the index."""
        if index >= len(self.__discovered_machines):
            raise ValueError(f"The provided index : {index} is higher than the discovered machines count : {len(self.__discovered_machines)}.")

        return self.__discovered_machines[index]
    
    def get_local_attacks_count(self) -> int:
        """Return the local attacks count."""
        return len(self.__local_attack_index)

    def get_remote_attacks_count(self) -> int:
        """Return the remote attacks count."""
        return len(self.__remote_attack_index)

    def get_discovered_machines(self) -> List[str]:
        """Return the discovered machines and their associated activity history."""
        return self.__discovered_machines
    
    def goals_description(self) -> str:
        """Describe the attacker goals."""
        return f"The attacker must at least capture {self.goals.nb_flag} flags and at least gather {self.goals.reward} rewards."
    
    def attack_as_string(self) -> Dict[int, str]:
        """Return the attacks as string by index."""
        return dict([(i, a.get_name()) for i, a in self.__attacks.items()])
    
    def get_cumulative_rewards(self) -> np.ndarray:
        """Return the cumulative reward."""
        return np.cumsum(np.array(self.__cumulative_reward))
    
    def add_new_properties(self, discovered_properties: List[str]) -> int:
        """Add new properties and return the number of unknown properties that have been discovered."""
        new_properties = [p for p in discovered_properties if p not in self.__found_properties]
        self.__found_properties += new_properties

        return len(new_properties)
    
    def get_total_reward(self) -> float:
        """Return the cumulative reward."""
        return self.__reward
    
    def set_positions(self, instance_names: List[str]) -> None:
        """Set machines where the attacker is initially connected."""
        for name in instance_names:

            self.__discovered_machines.append(name)
            self.__last_connections[name] = self.__start_time
            self.__attacks_tried[name] = dict()
        
    def is_discovered(self, machine: str):
        """Return whether a machine has been discovered or not."""
        return machine in self.__discovered_machines

    def get_discovered_credentials(self) -> List[Tuple[str, str, str]]:
        """Return the discovered credential."""
        return [cred.get_description() for cred in self.__found_credential]
    
    def update_history(self, time: float, reward: float, attack_name: str, machine_instance_name: str, result: str, type: str, flag: bool, step: int) -> None:
        """Update the attack history."""
        self.history['time'].append(time - self.__start_time)
        self.history['reward'].append(reward)
        self.history['attack name'].append(attack_name)
        self.history['machine instance name'].append(machine_instance_name)
        self.history['result'].append(result)
        self.history['type'].append(type)
        self.history['flag'].append(flag)
        self.history['iteration'].append(step)
        self.history['cumulative reward'].append(sum(self.history['reward']))

    def get_infected_machines(self) -> List[Tuple[str, float]]:
        """Return the infected machine instance name with the last connection time."""
        return [(instance_name, self.__last_connections[instance_name] - self.__start_time) for instance_name in self.__discovered_machines if self.__last_connections[instance_name]]
    
    def get_attack_outcome(self, attack: Attack, machine: Machine) -> Tuple[bool, float, bool]:
        """Return whether the attack has been performed successfully, the reward executing it on the machine and whether a flag has been captured or not."""
        instance_name = machine.get_instance_name()
        attack_id = attack.get_id()
        type_attack = attack.get_type()
        type = 'remote' if type_attack == ActionType.REMOTE else 'local'
        reward = 0

        if (attack_id not in [a.get_id() for a in self.__attacks_by_machine[instance_name]]) or (not machine.is_running()):

            reward += Reward.failed_attack

            if self.__attacks_name_to_id[attack.get_name()] not in self.__attacks_tried[instance_name]:

                self.__attacks_tried[instance_name][self.__attacks_name_to_id[attack.get_name()]] = False

            self.update_history(time.time(), reward, attack.get_name(), instance_name, 'failed', type, False, self.step_count)

            return False, reward, False

        attack_phase_names = attack.get_outcomes()
        
        if self.__attacks_name_to_id[attack.get_name()] in self.__attacks_tried[instance_name]:

            already_get_rewarded = self.__attacks_tried[instance_name][self.__attacks_name_to_id[attack.get_name()]]
        
        else:

            already_get_rewarded = False

        self.__attacks_tried[instance_name][self.__attacks_name_to_id[attack.get_name()]] = True

        for phase_name in attack_phase_names:

            outcomes = machine.get_outcome(phase_name)

            if already_get_rewarded:
                
                reward += Reward.repeat_attack
            
            for outcome in outcomes:

                if not already_get_rewarded:

                    reward += outcome.get_absolute_value()

                if isinstance(outcome, LeakedCredentials):

                    credentials, flag = outcome.get()

                    for credential in credentials:

                        if credential not in self.__found_credential:

                            self.__found_credential.append(credential)
                            machine_instance_name = credential.machine

                            if machine_instance_name not in self.__discovered_machines:

                                self.__discovered_machines.append(machine_instance_name)
                                self.__last_connections[machine_instance_name] = None
                                self.__attacks_tried[machine_instance_name] = dict()
                                reward += Reward.discovered_machineIP

                            reward += Reward.discovered_credential
                
                elif isinstance(outcome, LeakedMachineIP):

                    discovered_machines, flag = outcome.get()

                    for discovered_machine in discovered_machines:

                        if discovered_machine not in self.__discovered_machines:

                            self.__discovered_machines.append(discovered_machine)
                            self.__last_connections[discovered_machine] = None
                            self.__attacks_tried[discovered_machine] = dict()
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

        self.__attacks_tried[instance_name][self.__attacks_name_to_id[attack.get_name()]] = True
        self.update_history(time.time(), reward, attack.get_name(), instance_name, 'successfull', type, flag, self.step_count)

        return True, reward, flag
    
    def execute_local_action(self, attacker_action: np.ndarray) -> Tuple[float, bool, Activity]:
        """Execute a local action."""
        machine_instance_name = self.instance_name_by_index(attacker_action[0])        
        attack = self.__attacks[attacker_action[1]]

        if machine_instance_name not in self.__discovered_machines:
            raise ValueError(f"The machine {machine_instance_name} isn't discovered yet.")
        
        if attack.get_type() == ActionType.REMOTE:
            raise ValueError("To execute a local action, you need to used an attack with a local type.")
        
        if not self.__last_connections[machine_instance_name]:

            if self.__attacks_name_to_id[attack.get_name()] not in self.__attacks_tried[machine_instance_name]:

                self.__attacks_tried[machine_instance_name][self.__attacks_name_to_id[attack.get_name()]] = False

            reward = Reward.failed_attack

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

            return reward, flag, activity
        
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

        attack = self.__attacks[attacker_action[2]]

        if not self.__last_connections[source]:
            
            if self.__attacks_name_to_id[attack.get_name()] not in self.__attacks_tried[target]:

                self.__attacks_tried[target][self.__attacks_name_to_id[attack.get_name()]] = False

            return Reward.failed_attack, False, Activity()

        if attack.get_type() == ActionType.LOCAL:
            raise ValueError("To execute a remote action, you need to used an attack with a remote type.")

        target_machine = get_machines_by_name(target, self.__network.get_machine_list())[0]

        path = self.__network.get_path(source, target)

        ports_and_actions = [(target_machine.get_service_name(ds), ds) for ds in attack.get_data_sources() if target_machine.is_data_source_available(ds)]

        if len(ports_and_actions) == 0:

            reward = Reward.failed_attack
            self.__attacks_tried[target][self.__attacks_name_to_id[attack.get_name()]] = False
            self.update_history(time.time(), reward, attack.get_name(), target, 'failed', 'remote', False, self.step_count)

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

                if self.__attacks_name_to_id[attack.get_name()] not in self.__attacks_tried[target]:

                    self.__attacks_tried[target][self.__attacks_name_to_id[attack.get_name()]] = False

                self.update_history(time.time(), 0, attack.get_name(), target, 'network failed', 'remote', False, self.step_count)

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

                return reward, flag, activity

            return reward, flag, Activity()
    
    def connect(self, attacker_action: np.ndarray) -> Tuple[float, bool, Activity]:
        """Connect."""
        source = self.instance_name_by_index(attacker_action[0])
        credential = self.__found_credential[attacker_action[1]]

        if source not in self.__discovered_machines:
            raise ValueError(f"The machine {source} isn't discovered yet.")

        target = credential.machine

        if not self.__last_connections[source]:

            reward = Reward.failed_attack
            self.update_history(time.time(), reward, 'User Account Authentification', target, 'failed', 'connect', False, self.step_count)

            return reward, False, Activity()

        if target not in self.__discovered_machines:
            raise ValueError(f"The machine {target} isn't discovered yet.")
        
        service = credential.port
        path = self.__network.get_path(source, target)

        if isinstance(path, int):

            if self.__last_connections[target]:

                self.__last_connections[target] = time.time()
                reward = Reward.repeat_attack
                self.update_history(time.time(), reward, 'User Account Authentification', target, 'repeat', 'connect', False, self.step_count)

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

            if self.__attacks_name_to_id[credential.cred] not in self.__attacks_tried[target]:

                self.__attacks_tried[target][self.__attacks_name_to_id[credential.cred]] = False

            self.update_history(time.time(), 0, 'User Account Authentification', target, 'network failed', 'connect', False, self.step_count)

            return 0, False, activity

        reward = 0
        flag = False

        if not self.__last_connections[target]:
            
            self.__last_connections[target] = time.time()
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

        self.__attacks_tried[target][self.__attacks_name_to_id[credential.cred]] = True
        self.update_history(time.time(), reward, 'User Account Authentification', target, 'successfull', 'connect', flag, self.step_count)
        
        return reward, flag, activity
        
    def on_step(self, attacker_action: Dict[str, np.ndarray]) -> Tuple[float, Activity]:
        """Return the attacker activity."""
        if 'connect' in attacker_action:
            
            reward, flag, activity = self.connect(attacker_action['connect'])

        elif 'submarine' in attacker_action:

            return Reward.waiting, Activity()
        
        elif 'local' in attacker_action:

            reward, flag, activity = self.execute_local_action(attacker_action['local'])
        
        elif 'remote' in attacker_action:

            reward, flag, activity = self.execute_remote_action(attacker_action['remote'])
        
        if flag:

            self.__captured_flag += 1

        self.__reward += reward
        self.step_count += 1
        
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
        self.__discovered_machines: List[str] = []
        self.__last_connections: Dict[str, float] = dict()
        self.__attacks_tried: Dict[str, Dict[int, float]] = dict()
        self.__network.reset()
        self.__start_time = new_start_time
        self.__captured_flag = 0
        self.__reward = 0
        self.__found_properties: List[str] = []
        self.step_count = 1        
        self.history: Dict[str, List[object]] = {
            'time': [],
            'iteration': [],
            'attack name': [],
            'machine instance name': [],
            'reward': [],
            'result': [],
            'type': [],
            'flag': [],
            'cumulative reward': []
        }

        self.set_positions([m.get_instance_name() for m in self.__network.get_machine_list() if m.is_infected])
    
    def display_history(self, x='iteration') -> None:
        """Display the attacker action history through a time line."""
        df = pd.DataFrame(self.history, columns=['time', 'iteration', 'reward', 'attack name', 'machine instance name', 'result', 'type', 'flag', 'cumulative reward'])
        fig = px.scatter(df, x=x, y='cumulative reward', color='result', symbol='type', title="Attacker history", hover_data=['time', 'iteration', 'reward', 'attack name', 'machine instance name', 'type', 'flag'])
        #fig.update_traces(mode="markers+lines")
        fig.show()
    
    def get_attacker_history(self) -> Dict[str, object]:
        """Return the attacker attack history."""
        return self.history
    
    def sample_random_valid_action(self, spaces: DiscriminateSpaces) -> Dict[str, np.ndarray]:
        """Sample a random action that the attacker is able to perform."""
        infected_machine_index = []

        for i, last_connection in enumerate(self.__last_connections):
            
            if last_connection:

                infected_machine_index.append(i)
        
        discovered_machine_count = len(self.__discovered_machines)
        gathered_creddentials_count = len(self.__found_credential)

        range_index = ['connect', 'submarine', 'local', 'remote']

        if gathered_creddentials_count == 0:

            range_index.remove('connect')
        
        if ( len(infected_machine_index) == 1 ) and ( discovered_machine_count == 1 ):

            range_index.remove('remote')

        action_type = spaces.sample_action_type(range_index)

        if action_type == 'connect':

            source_machine_index = np.random.choice(infected_machine_index)
            cred_index = np.random.randint(0, gathered_creddentials_count)
        
            return {'connect': np.array([source_machine_index, cred_index])}

        if action_type == 'remote':

            source_machine_index = np.random.choice(infected_machine_index)
            target_machine_candidates = [i for i in range(discovered_machine_count) if i != source_machine_index]
            target_machine_index = np.random.choice(target_machine_candidates)
            attack_index = np.random.choice(self.__remote_attack_index)
        
            return {'remote': np.array([source_machine_index, target_machine_index, attack_index])}

        if action_type == 'submarine': 

            return {'submarine': np.array([np.random.choice(infected_machine_index)])}
        
        if action_type == 'local':

            machine_index = np.random.choice(infected_machine_index)
            attack_index = np.random.choice(self.__local_attack_index)
        
            return {'local': np.array([machine_index, attack_index])}
