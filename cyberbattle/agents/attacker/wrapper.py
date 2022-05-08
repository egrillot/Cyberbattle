

import numpy as np

from typing import List
from gym import spaces

from .attacker_interface import Attacker
from..battle_environment import AttackerBounds


class Feature(spaces.MultiDiscrete):
    """Feature class.
    
    This class will allow us to wrap the observations into a constant dimensional vector for the model.
    """

    def __init__(self, nvec: List[int], name: str, type: str, bounds: AttackerBounds) -> None:
        """Init.
        
        Input:
        nvec: the values range for each vector components (List[int])
        name: the feature name (str)
        type: if 'local', it means the feature need to be gathered from a machine else enter type as 'global' (str).
        """
        if type not in ['global', 'local']:
            raise ValueError(f"The provided type {type} isn't defined, please choose among the following types : 'global' or 'local'.")

        self.dim = len(nvec)
        self.name = name
        self.type = type
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

    def get_vector(self, attacker: Attacker, machine_index: int) -> np.ndarray:
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
        type = 'local'
        super().__init__(nvec, name, type, bounds, bounds)
    
    def get_vector(self, attacker: Attacker, machine_index: int) -> np.ndarray:
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
        type = 'global'
        super().__init__(nvec, name, type, bounds)
    
    def get_vector(self, attacker: Attacker, machine_index: int) -> np.ndarray:
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
        type = 'global'
        super().__init__(nvec, name, type, bounds)

    def get_vector(self, attacker: Attacker, machine_index: int) -> np.ndarray:
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
        type = 'global'
        super().__init__(nvec, name, type, bounds)

    def get_vector(self, attacker: Attacker, machine_index: int) -> np.ndarray:
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
        type = 'global'
        super().__init__(nvec, name, type, bounds)
    
    def get_vector(self, attacker: Attacker, machine_index: int) -> np.ndarray:
        """Return the discovered machines not infected count."""
        discovered_machines_count = len(attacker.get_discovered_machines())
        infected_machines_count = len(attacker.get_infected_machines())

        if discovered_machines_count - infected_machines_count < 0:
            raise ValueError(f"The infected machines count {infected_machines_count} is higher than the discovered machines count : {discovered_machines_count}.")
        
        return np.array([discovered_machines_count - infected_machines_count], dtype=int)


class ActionsTriedAt(Feature):
    """ActionsTriedAt class.
    
    It allows the model to check how much each attacks have been tried on a provided machine.
    """

    def __init__(self, bounds: AttackerBounds) -> None:
        """Init."""
        name = 'actions tried at count'
        nvec = [2] * ( bounds.maximum_local_attack + bounds.maximum_remote_attack + 1 )
        type = 'local'
        super().__init__(nvec, name, type, bounds)
    
    def get_vector(self, attacker: Attacker, machine_index: int) -> np.ndarray:
        """Return how much each attacks have been tried on a provided machine."""
        machine_instance_name = attacker.instance_name_by_index(machine_index)
        vector = np.zeros((self.bounds.maximum_local_attack + self.bounds.maximum_remote_attack + 1, ), dtype=int)

        tracker = attacker.get_machine_tracker(machine_instance_name)

        if tracker.last_connection > 0:

            vector[-1] = 1
        
        vector[np.array(list(tracker.attacks.keys()))] = 1

        return vector


class AttacksSuccessfull(Feature):
    """AttacksSuccessfull class.
    
    It allows the model to get how much attacks have been performed successfully on a provided machine during the last 100 steps.
    """

    window = 100

    def __init__(self, bounds: AttackerBounds) -> None:
        """Init."""
        name = 'successfull attacks count'
        nvec = [self.window] * ( bounds.maximum_local_attack + bounds.maximum_remote_attack + 1 )
        type = 'local'
        super().__init__(nvec, name, type, bounds)

    def get_vector(self, attacker: Attacker, machine_index: int) -> np.ndarray:
        """Return how much attacks have been performed successfully on the provided machine."""
        machine_instance_name = attacker.instance_name_by_index(machine_index)

        return attacker.successfull_actions_count(machine_instance_name, self.window)


class AttacksFailed(Feature):
    """AttacksFailed class.
    
    It allows the model to get how much attacks have been performed failed on a provided machine during the last 100 steps.
    """

    window = 100

    def __init__(self, bounds: AttackerBounds) -> None:
        """Init."""
        name = 'failed attacks count'
        nvec = [self.window] * ( bounds.maximum_local_attack + bounds.maximum_remote_attack + 1 )
        type = 'local'
        super().__init__(nvec, name, type, bounds)

    def get_vector(self, attacker: Attacker, machine_index: int) -> np.ndarray:
        """Return how much attacks have been performed failed on the provided machine."""
        machine_instance_name = attacker.instance_name_by_index(machine_index)

        return attacker.failed_actions_count(machine_instance_name, self.window)

