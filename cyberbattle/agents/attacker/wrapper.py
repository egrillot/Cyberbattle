

import numpy as np

from typing import List, Dict, Tuple
from gym import spaces

from .attacker_interface import Attacker
from..battle_environment import AttackerBounds, CyberBattleEnv


predefined_features = [
    'machine discovery',
    'discovered credentials count',
    'infected machines count',
    'discovered machine count',
    'discovered machine not infected',
    'actions tried at count',
    'succesful attacks count',
    'failed attacks count',
    'All'
]


class ActionTracker:
    """ActionTracker class.
    
    This class allows us to count every actions made by the attacker precising whether these attacks were performed succesfully or not.
    """

    def __init__(self, bounds: AttackerBounds) -> None:
        """Init."""
        self.history_actions = np.zeros((bounds.maximum_machines_count, bounds.maximum_local_attack + bounds.maximum_remote_attack + bounds.maximum_credentials_count, 2), dtype=int)
        self.bounds = bounds
    
    def procces_result(self, attacker_action: Dict[str, np.ndarray], reward: float) -> None:
        """Account for what happened."""
        index = 0 if reward > 0 else 1

        if 'connect' in attacker_action:

            attack = attacker_action['connect']
            self.history_actions[attack[0], self.bounds.maximum_local_attack + self.bounds.maximum_remote_attack + attack[1], index] += 1

        elif 'remote' in attacker_action:

            attack = attacker_action['remote']
            self.history_actions[attack[1], attack[2], index] += 1

        elif 'local' in attacker_action:

            attack = attacker_action['local']
            self.history_actions[attack[0], attack[1], index] += 1
    
    def get_tracking_succesful_actions(self, machine_index: int) -> np.ndarray:
        """Return the tracking on the provided machine index of the succesful actions."""
        return self.history_actions[machine_index, :, 0]
    
    def get_tracking_failed_actions(self, machine_index: int) -> np.ndarray:
        """Return the tracking on the provided machine index of the failed actions."""
        return self.history_actions[machine_index, :, 1]
    
    def reset(self) -> None:
        """Reset the tracker."""
        self.history_actions = np.zeros((self.bounds.maximum_machines_count, self.bounds.maximum_local_attack + self.bounds.maximum_remote_attack + self.bounds.maximum_credentials_count, 2), dtype=int)


class Feature(spaces.MultiDiscrete):
    """Feature class.
    
    This class will allow us to wrap the observations into a constant dimensional vector for the model.
    """

    def __init__(self, nvec: List[int], name: str, bounds: AttackerBounds) -> None:
        """Init.
        
        Input:
        nvec: the values range for each vector components (List[int])
        name: the feature name (str).
        """
        self.dim = len(nvec)
        self.name = name
        self.bounds = bounds
        super().__init__(nvec)
    
    def flat_size(self) -> int:
        """Return the vector flat size (number of writable vectors)."""
        return np.prod(self.nvec)
    
    def dimension(self) -> int:
        """Return the vector dimension."""
        return self.dim

    def get_name(self) -> str:
        """Return the feature name."""
        return self.name

    def get_vector(self, attacker: Attacker, machine_index: int, tracker: ActionTracker) -> np.ndarray:
        """Return the feature vector that describes the wanted information on the provided machine_index (but can be global too) with respect to the attacker discovery."""
        raise NotImplementedError


class MachineDiscovery(Feature):
    """MachineDiscovery class.
    
    It allows the model to know how recently the machine has been discovered.
    """

    def __init__(self, bounds: AttackerBounds) -> None:
        """Init."""
        name = 'machine discovery'
        nvec = [bounds.maximum_machines_count]
        super().__init__(nvec, name, bounds)
    
    def get_vector(self, attacker: Attacker, machine_index: int, tracker: ActionTracker) -> np.ndarray:
        """Return how recently the machine has been discovered by giving its index."""
        discovered_machines_count = len(attacker.get_discovered_machines())

        if discovered_machines_count <=  machine_index:
            raise ValueError(f"The provided index : {machine_index} is higher than the number of discovered machine : {discovered_machines_count}.")

        return np.array([machine_index], dtype=int)


class CredentialDiscovery(Feature):
    """CredentialDiscovery class.
    
    It allows the model to know how much credentials have been discovered yet.
    """

    def __init__(self, bounds: AttackerBounds) -> None:
        """Init."""
        name = 'discovered credentials count'
        nvec = [bounds.maximum_credentials_count]
        super().__init__(nvec, name, bounds)
    
    def get_vector(self, attacker: Attacker, machine_index: int, tracker: ActionTracker) -> np.ndarray:
        """Return the discovered credentials count."""
        return np.array([len(attacker.get_discovered_credentials())], dtype=int)


class InfectedMachines(Feature):
    """InfectedMachines class.
    
    It allows the model to see how much machines are infected from its point of view.
    Indeed, the defender may have reimaged a machine and the attacker still doesn't know he's not anymore connected.
    """

    def __init__(self, bounds: AttackerBounds) -> None:
        """Init."""
        name = 'infected machines count'
        nvec = [bounds.maximum_machines_count]
        super().__init__(nvec, name, bounds)

    def get_vector(self, attacker: Attacker, machine_index: int, tracker: ActionTracker) -> np.ndarray:
        """Return the infected machines count."""
        return np.array([len(attacker.get_infected_machines())], dtype=int)


class DiscoveredMachine(Feature):
    """DiscoveredMachine class.
    
    It allows the model to see how much machine have been discovered yet.
    """

    def __init__(self, bounds: AttackerBounds) -> None:
        """Init."""
        name = 'discovered machine count'
        nvec = [bounds.maximum_machines_count]
        super().__init__(nvec, name, bounds)

    def get_vector(self, attacker: Attacker, machine_index: int, tracker: ActionTracker) -> np.ndarray:
        """Return the discovered machines count."""
        return np.array([len(attacker.get_discovered_machines())], dtype=int)


class DiscoveredMachinesNotInfected(Feature):
    """DiscoveredMachinesNotInfected class.
    
    It allows the model to know how much discovered machines are not infected.
    Again it is from its point of view.
    """

    def __init__(self, bounds: AttackerBounds) -> None:
        """Init."""
        name = 'discovered machine not infected'
        nvec = [bounds.maximum_machines_count]
        super().__init__(nvec, name, bounds)
    
    def get_vector(self, attacker: Attacker, machine_index: int, tracker: ActionTracker) -> np.ndarray:
        """Return the discovered machines not infected count."""
        discovered_machines_count = len(attacker.get_discovered_machines())
        infected_machines_count = len(attacker.get_infected_machines())

        if discovered_machines_count - infected_machines_count < 0:
            raise ValueError(f"The infected machines count {infected_machines_count} is higher than the discovered machines count : {discovered_machines_count}.")
        
        return np.array([discovered_machines_count - infected_machines_count], dtype=int)


class ActionsTriedAt(Feature):
    """ActionsTriedAt class.
    
    It allows the model to check if ann attack has been tried on a provided machine.
    """

    def __init__(self, bounds: AttackerBounds) -> None:
        """Init."""
        name = 'actions tried at count'
        nvec = [2] * ( bounds.maximum_local_attack + bounds.maximum_remote_attack + bounds.maximum_credentials_count )
        super().__init__(nvec, name, bounds)
    
    def get_vector(self, attacker: Attacker, machine_index: int, tracker: ActionTracker) -> np.ndarray:
        """Return if ann attack has been tried on a provided machine."""
        failed_actions_tracking = tracker.get_tracking_failed_actions(machine_index) 
        succesful_actions_tracking = tracker.get_tracking_succesful_actions(machine_index)

        return np.where(failed_actions_tracking + succesful_actions_tracking != 0, 1, 0)


class Attackssuccesful(Feature):
    """Attackssuccesful class.
    
    It allows the model to get how much attacks have been performed succesfully on a provided machine.
    """

    def __init__(self, bounds: AttackerBounds, max_count: int=100) -> None:
        """Init."""
        name = 'succesful attacks count'
        self.max_count = max_count
        nvec = [max_count] * ( bounds.maximum_local_attack + bounds.maximum_remote_attack + bounds.maximum_credentials_count )
        super().__init__(nvec, name, bounds)

    def get_vector(self, attacker: Attacker, machine_index: int, tracker: ActionTracker) -> np.ndarray:
        """Return how much attacks have been performed succesfully on the provided machine.
        
        The output is a clipped vector (min=0, max=self.max_count).
        """
        succesful_actions = tracker.get_tracking_succesful_actions(machine_index)

        return np.clip(succesful_actions, 0, self.max_count)


class AttacksFailed(Feature):
    """AttacksFailed class.
    
    It allows the model to get how much attacks have been performed failed on a provided machine.
    """

    def __init__(self, bounds: AttackerBounds, max_count: int=100) -> None:
        """Init."""
        name = 'failed attacks count'
        self.max_count = max_count
        nvec = [max_count] * ( bounds.maximum_local_attack + bounds.maximum_remote_attack + bounds.maximum_credentials_count )
        super().__init__(nvec, name, bounds)

    def get_vector(self, attacker: Attacker, machine_index: int, tracker: ActionTracker) -> np.ndarray:
        """Return how much attacks have been performed failed on the provided machine.
        
        The output is a clipped vector (min=0, max=self.max_count).
        """
        failed_actions = tracker.get_tracking_failed_actions(machine_index)

        return np.clip(failed_actions, 0, self.max_count)


class ConcatenateFeatures(Feature):
    """ConcatenateFeatures class.
    
    This class allows us to concatenate different features in a single vector.
    """

    def __init__(self, bounds: AttackerBounds, features: List[str] or List[Feature]) -> None:
        """Init."""
        name = 'concatenate features'

        if sum([1 for f in features if isinstance(f, str)]) == len(features):

            for f in features:

                if f not in predefined_features:
                    raise ValueError(f"Undefined features name : {f}, please select features among the following names : {predefined_features}.")

            all_features: List[Feature] = [
                MachineDiscovery(bounds),
                CredentialDiscovery(bounds),
                InfectedMachines(bounds),
                DiscoveredMachine(bounds),
                DiscoveredMachinesNotInfected(bounds),
                ActionsTriedAt(bounds),
                Attackssuccesful(bounds),
                AttacksFailed(bounds)
            ]

            self.features: List[Feature] = []

            if 'All' in features:

                self.features = all_features
            
            else:

                for feature_name in features:

                    self.features.append([f for f in all_features if f.get_name() == feature_name][0])
        
        elif sum([1 for f in features if isinstance(f, Feature)]) == len(features):

            self.features: List[Feature] = features
        
        else:
            raise ValueError(f"Please provide a list of ")

        nvec = np.concatenate([f.nvec for f in self.features])
            
        super().__init__(nvec, name, bounds)
    
    def get_vector(self, attacker: Attacker, machine_index: int, tracker: ActionTracker) -> np.ndarray:
        """Return the concatenation of vectors."""
        vectors = []

        for feature in self.features:
            
            vectors.append(feature.get_vector(attacker, machine_index, tracker))
        
        return np.concatenate(vectors, axis=-1)


class Wrapper:
    """Wrapper class.
    
    This class will manage model interaction with the environment. Specifically, this class allows us to wrap the environment
    into features concatenation and to convert the model decision to an action of type Dict[str, np.ndarray].
    """

    def __init__(self, features: List[str] or List[Feature], bounds: AttackerBounds, hash_size: int=10000) -> None:
        """Init.
        
        Inputs:
        features: the list of desired features the model will be able to use to watch the environment (List[str])
        bounds: the attacker environment bounds (AttackerBounds).
        """
        self.features = ConcatenateFeatures(bounds, features)
        self.tracker = ActionTracker(bounds)
        self.maximum_machine_count = bounds.maximum_machines_count
        self.local_action_count = bounds.maximum_local_attack
        self.remote_action_count = bounds.maximum_remote_attack
        self.credential_action_count = bounds.maximum_credentials_count
        self.hash_size = hash_size

        self.reset()
    
    def states_flat_size(self) -> int:
        """Return how many state can be represented."""
        return self.hash_size
    
    def reset(self) -> None:
        """Reset the wrapper."""
        self.tracker.reset()
    
    def process_result(self, reward: float, action: Dict[str, np.ndarray]) -> None:
        """Process what happened in the environment."""
        self.tracker.procces_result(reward=reward, attacker_action=action)

    def model2action(self, action: int, source: int) -> Dict[str, np.ndarray]:
        """Return an action readable in the environment from chosen actions by the model."""
        if action == 0:

            return {'submarine': np.array([source])}

        action -= 1

        if action < self.local_action_count:

            return {'local': np.array([source, action])}
        
        action -= self.local_action_count
        
        if action < self.maximum_machine_count * self.remote_action_count:

            target = action // self.remote_action_count
            action = action % self.remote_action_count

            return {'remote': np.array([source, target, action])}
        
        action -= self.maximum_machine_count * self.remote_action_count

        return {'connect': np.array([source, action])}    

    def action2model(self, action: Dict[str, np.ndarray]) -> Tuple[int, int]:
        """Return models actions and chosen that could lead us to the provided action."""
        if 'submarine' in action:

            return 0, action['submarine'][0]
        
        elif 'local' in action:

            return action['local'][1] + 1, action['local'][0]
        
        elif 'remote' in action:

            return action['remote'][1] * self.remote_action_count + action['remote'][2] + 1 - self.local_action_count, action['remote'][0]
        
        elif 'connect' in action:

            return action['connect'][1] + self.remote_action_count * self.maximum_machine_count + 1 + self.local_action_count, action['connect'][0]
    
    def observation(self, env: CyberBattleEnv) -> np.ndarray:
        """Return an array of features applied on each discovered machines."""
        attacker_interface = env.get_attacker()
        discovered_machine_count = len(attacker_interface.get_discovered_machines())
        vectors = np.zeros((discovered_machine_count, self.features.nvec.shape[0]), dtype=int)
        
        for i in range(discovered_machine_count):

            vectors[i, :] = self.features.get_vector(attacker_interface, i, self.tracker)

        return vectors
    
    def encode_vector(self, feature_vector: np.ndarray) -> int:
        """Return an encoding integer wich will correspond to the state index."""
        return hash(str(feature_vector.tolist())) % self.hash_size
    
    def encode_vectors(self, feature_vectors: np.ndarray) -> List[int]:
        """Encode each vectors assuming the number of rows corresponds to the number of vector."""
        encoded_vectors = []
        for i in range(feature_vectors.shape[0]):

           encoded_vectors.append(self.encode_vector(feature_vectors[i, :]))
        
        return encoded_vectors
    
    def states_observed(self, env: CyberBattleEnv) -> List[int]:
        """Return a list of encoded features applied on each discovered machines."""
        return self.encode_vectors(self.observation(env))
    
    def states_observed_at(self, env: CyberBattleEnv, source: int) -> int:
        """Return the encoded features applied on the provided discovered machine."""
        feature = self.features.get_vector(env.get_attacker(), source, self.tracker)

        return self.encode_vector(feature)
