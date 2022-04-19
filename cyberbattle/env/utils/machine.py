"""Provide classes to define network components, all classes inherit from the Machine class."""


from typing import List
from ...vulnerabilities import outcomes
from . import flow

class Machine:
    """Abstract class to define a machine in the environment."""

    def __init__(
        self,
        name: str,
        instance_name: str,
        platforms: List[str],
        connected_machines: List[int],
        url_image: str,
        outcomes: List[outcomes.Outcome]=[],
        value: int=0,
        is_infected: bool=False,
    ):
        """Init.
        
        Input:
        ip_adress: ip adress of the machine (int)
        name: str
        instance_name: name to define the launched instance (str)
        platforms: list of systems operating on the machine (List[str])
        connected_machines: set of int corresponding to connected machine ip adresses (List[str])
        url_image: image url to display the machine (str)
        outcomes: list of possible results of attacks that the attacker can carry out on the machine (List[Outcome])
        value: integer defining the machine importance in the network (int)
        is_infected: if True, it means that the attacker is connected as a local user on the machine at the simulation start (bool).
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
    
    def get_outcomes(self) -> List[outcomes.Outcome]:
        """Return outcome list."""
        return self.outcomes


def get_machines_by_name(name: str, machines: List[Machine]) -> List[Machine]:
    """Return a list of machines whose name corresponds to the entry."""
    return [m for m in machines if m.get_name() == name]


class Plug(Machine):
    """Connector for linking several machines."""

    def __init__(self, instance_name: str, platforms: List[str], connected_machines: List[int], outcomes: List[outcomes.Outcome]=[], value: int=0, is_infected: bool=False):
        """Init the connected machine to the plug."""

        if len(connected_machines) <= 1:
            raise ValueError('A plug must at least link 2 machines instead of {}'.format(len(connected_machines)))

        name = 'Plug'
        url_image = './env/images/Switch.jpg'
        super().__init__(name, instance_name, platforms, connected_machines, url_image, outcomes, value, is_infected)


class Firewall(Machine):
    """Firewall class."""

    def __init__(self, incomings: List[flow.Traffic], outgoings: List[flow.Traffic], instance_name: str, platforms: List[str], connected_machines: List[int], outcomes: List[outcomes.Outcome]=[], value: int=0, is_infected: bool=False):
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
        url_image = './env/images/Firewall.jpg'
        super().__init__(name, instance_name, platforms, connected_machines, url_image, outcomes, value, is_infected)
    
    def is_passing(self, port_name: str, coming_from: Machine) -> bool:
        """Check if the providing request can pass the firewall in the providing direction.
        
        Inputs: 
        port_name: the requested port to pass (str)
        traffic: direction of traffic (TrafficDirection)
        Output: bool.
        """
        ip_address = coming_from.get_ip_adress()
        if ip_address not in [m.get_ip_adress() for m in self.connected_machines]:
            raise ValueError("The firewall {} isn't connecting provided machine: {}".format(self.ip_address, ip_address))
        
        traffic_rules = self.outgoings if ip_address == self.connected_machines[0].get_ip_adress() else self.incomings

        for traffic in traffic_rules:

            if traffic.port == port_name and traffic.rule == outcomes.Rule.ALLOWED:

                return True
        
        return False


class Client(Machine):
    """Client class."""

    def __init__(self, instance_name: str, platforms: List[str], connected_machines: List[int], outcomes: List[outcomes.Outcome]=[], value: int=0, is_infected: bool=False):
        """Init."""
        name = 'PC'
        url_image = './env/images/PC.png'
        super().__init__(name, instance_name, platforms, connected_machines, url_image, outcomes, value, is_infected)


class Server(Machine):
    """Server class."""

    def __init__(self, instance_name: str, platforms: List[str], connected_machines: List[int], outcomes: List[outcomes.Outcome]=[], value: int=0, is_infected: bool=False):
        """Init."""
        name = 'Server'
        url_image = './env/images/Server.png'
        super().__init__(name, instance_name, platforms, connected_machines, url_image, outcomes, value, is_infected)


class Cloud(Machine):
    """Clourd class (external servers)."""

    def __init__(self, instance_name: str, platforms: List[str], connected_machines: List[int], outcomes: List[outcomes.Outcome]=[], value: int=0, is_infected: bool=False):
        """Init."""
        name = 'Cloud'
        url_image = './env/images/Cloud.jpg'
        super().__init__(name, instance_name, platforms, connected_machines, url_image, outcomes, value, is_infected)
