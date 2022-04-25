"""Provide classes to define network components, all classes inherit from the Machine class."""


from typing import List, Dict, Tuple
from .data import Data_source
from ...vulnerabilities.outcomes import Outcome
from .flow import Traffic, Rule

class Machine:
    """Abstract class to define a machine in the environment."""

    def __init__(
        self,
        name: str,
        instance_name: str,
        platforms: List[str],
        connected_machines: List[str],
        url_image: str,
        outcomes: List[Outcome]=[],
        value: int=0,
        is_infected: bool=False,
        data_sources: Dict[str, List[Data_source]]=dict()
    ):
        """Init.
        
        Input:
        ip_adress: ip adress of the machine (int)
        name: (str)
        instance_name: name to define the launched instance (str)
        platforms: list of systems operating on the machine (List[str])
        connected_machines: set of int corresponding to connected machine ip adresses (List[str])
        url_image: image url to display the machine (str)
        outcomes: list of possible results of attacks that the attacker can carry out on the machine (List[Outcome])
        value: integer defining the machine importance in the network (int)
        is_infected: if True, it means that the attacker is connected as a local user on the machine at the simulation start (bool)
        data_sources: dictionary of services offered by the machine linked to accessible data sources (Dict[str, List[Data_source]]).
        """
        self.ip_adress = None
        self.outcomes = outcomes
        self.name = name
        self.instance_name = instance_name
        self.platforms = platforms
        self.connected_machines = set(connected_machines)
        self.url_image = url_image
        self.value = value
        self.is_infected = is_infected
        self.service_to_data_sources = data_sources
    
    def set_ip_adress(self, ip_adress: int) -> None:
        """Set ip adress."""
        self.ip_adress = ip_adress

    def get_ip_adress(self) -> int:
        """Return the machine ip adress."""
        if self.ip_adress is None:
            raise ValueError("The ip adress hasn't yet been set.")

        return self.ip_adress
    
    def get_name(self) -> str:
        """Return machine name."""
        return self.name
    
    def get_instance_name(self) -> str:
        """Return the instance name."""
        return self.instance_name
    
    def get_platforms(self) -> List[str]:
        """Return operating systems on the machine."""
        return self.platforms
    
    def get_connected_machines(self) -> List[str]:
        """Return connected machines."""
        return self.connected_machines
    
    def get_url_image(self) -> str:
        """Return the url image."""
        return self.url_image

    def get_value(self) -> str:
        """Return the machine value."""
        return self.value
    
    def get_outcomes(self) -> List[Outcome]:
        """Return outcome list."""
        return self.outcomes
    
    def get_data_sources(self) -> Dict[str, List[Data_source]]:
        """Return the data sources."""
        return self.service_to_data_sources
    
    def is_data_source_available(self, data_source: str) -> bool:
        """Return if the data_source is available or not."""
        data_sources = self.service_to_data_sources.values()

        for ds in data_sources:

            if data_source in ds:

                return True
        
        return False
    
    def get_service_name(self, data_source: str) -> str:
        """Return the service name providing the data_source."""
        if not self.is_data_source_available(data_source):
            raise ValueError("The provided data source {} is not available on this machine.".format(data_source))
        
        for service, data_sources in self.service_to_data_sources.items():

            if data_source in data_sources:

                return service

class Plug(Machine):
    """Connector for linking several machines."""

    def __init__(self, instance_name: str, platforms: List[str], connected_machines: List[int], outcomes: List[Outcome]=[], value: int=0, is_infected: bool=False, data_sources: Dict[str, List[Data_source]]=dict()):
        """Init the connected machine to the plug."""

        if len(connected_machines) <= 1:
            raise ValueError('A plug must at least link 2 machines instead of {}'.format(len(connected_machines)))

        name = 'Plug'
        url_image = 'Switch.jpg'
        super().__init__(name, instance_name, platforms, connected_machines, url_image, outcomes, value, is_infected, data_sources)


class Firewall(Machine):
    """Firewall class."""

    def __init__(self, incomings: List[Traffic], outgoings: List[Traffic], instance_name: str, platforms: List[str], connected_machines: List[int], outcomes: List[Outcome]=[], value: int=0, is_infected: bool=False, data_sources: Dict[str, List[Data_source]]=dict()):
        """Init incoming and outgoing traffic rules.
        
        Inputs:
        incomings: list of incomings traffic rules
        outgoings: list of outgoings traffic rules.
        """
        if len(connected_machines) != 2:
            raise ValueError('A firewall must exactly link 2 machines instead of {}'.format(len(connected_machines)))

        self.incomings = incomings
        self.outgoings = outgoings
        name = 'Firewall'
        url_image = 'Firewall.jpg'
        super().__init__(name, instance_name, platforms, connected_machines, url_image, outcomes, value, is_infected, data_sources)
    
    def is_passing(self, port_name: str, coming_from: Machine) -> bool:
        """Check if the providing request can pass the firewall in the providing direction.
        
        Inputs: 
        port_name: the requested port to pass (str)
        traffic: direction of traffic (TrafficDirection)
        Output: bool.
        """
        instance_name = coming_from.get_instance_name()
        if instance_name not in [m for m in self.connected_machines]:
            raise ValueError("The firewall {} isn't connecting provided machine: {}".format(self.instance_name, instance_name))
        
        traffic_rules = self.outgoings if instance_name == self.connected_machines[0] else self.incomings

        for traffic in traffic_rules:

            if traffic.port == port_name and traffic.rule == Rule.ALLOWED:

                return True
        
        return False


class Client(Machine):
    """Client class."""

    def __init__(self, instance_name: str, platforms: List[str], connected_machines: List[int], outcomes: List[Outcome]=[], value: int=0, is_infected: bool=False, data_sources: Dict[str, List[Data_source]]=dict()):
        """Init."""
        name = 'PC'
        url_image = 'PC.png'
        super().__init__(name, instance_name, platforms, connected_machines, url_image, outcomes, value, is_infected, data_sources)


class Server(Machine):
    """Server class."""

    def __init__(self, instance_name: str, platforms: List[str], connected_machines: List[int], outcomes: List[Outcome]=[], value: int=0, is_infected: bool=False, data_sources: Dict[str, List[Data_source]]=dict()):
        """Init."""
        name = 'Server'
        url_image = 'Server.png'
        super().__init__(name, instance_name, platforms, connected_machines, url_image, outcomes, value, is_infected, data_sources)


class Cloud(Machine):
    """Clourd class (external servers)."""

    def __init__(self, instance_name: str, platforms: List[str], connected_machines: List[int], outcomes: List[Outcome]=[], value: int=0, is_infected: bool=False, data_sources: Dict[str, List[Data_source]]=dict()):
        """Init."""
        name = 'Cloud'
        url_image = 'Cloud.jpg'
        super().__init__(name, instance_name, platforms, connected_machines, url_image, outcomes, value, is_infected, data_sources)


def get_machines_by_name(name: str, machines: List[Machine]) -> List[Machine]:
    """Return a list of machines whose instance name corresponds to the entry name."""
    return [m for m in machines if m.get_instance_name() == name]


def firewall_instances(path: List[Machine]) -> List[Tuple[Machine, Firewall]]:
    """Return the list of firewalls in the path with the machine from where the traffic is coming."""
    firewalls = []
    for i, m in enumerate(path):

        if isinstance(m, Firewall):

            firewalls.append((path[i-1], m))
    
    return firewalls