"""Provide classes to define network components, all classes inherit from the Machine class."""


from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from copy import deepcopy
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
        data_sources: Dict[str, Dict[str, List[str]]]=dict()
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
        data_sources: dictionary of services offered by the machine linked to accessible data sources for each profile (Dict[str, Dict[str, List[str]]]).
        """
        self.ip_adress = None
        self.outcomes = outcomes
        self.name = name
        self.instance_name = instance_name
        self.platforms = platforms
        self.connected_machines = list(set(connected_machines))
        self.url_image = url_image
        self.value = value
        self.is_infected = is_infected
        self.__is_infected = deepcopy(is_infected)
        self.service_to_data_sources = data_sources
        self.incoming_history: Dict[str, Tuple[list[float], list[str]]] = {s: ([], []) for s in self.get_services()}
    
    def get_services(self) -> List[str]:
        """Return running services."""
        services = []
        for service_to_datas in self.service_to_data_sources.values():

            for port in service_to_datas.keys():

                services.append(port)
        
        return list(set(services))
    
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
    
    def get_data_sources(self) -> Dict[str, Dict[str, List[str]]]:
        """Return the data sources."""
        return self.service_to_data_sources
    
    def is_data_source_available(self, data_source: str) -> bool:
        """Return if the data_source is available or not."""
        for data_sources_per_service in self.service_to_data_sources.values():

            for data_sources in data_sources_per_service.values():

                if data_source in data_sources:

                    return True
        
        return False
    
    def get_service_name(self, data_source: str) -> str:
        """Return the service name providing the data_source."""
        if not self.is_data_source_available(data_source):
            raise ValueError("The provided data source {} is not available on this machine.".format(data_source))
        
        for data_sources_per_service in self.service_to_data_sources.values():

            for service, data_sources in data_sources_per_service.items():

                if data_source in data_sources:

                    return service
    
    def execute(self, profile: str, data_source: str) -> Tuple[bool, str]:
        """Return whether the profile can be retrieved by triggering the source data or not and if he can, it returns also by which port the service is provided."""
        if self.is_data_source_available(data_source):

            if profile in self.service_to_data_sources:

                for port, data_sources in self.service_to_data_sources[profile].items():

                    if data_source in data_sources:

                        return True, port
        
        return False, None

    def get_available_datasources_profile(self, profile: str) -> List[str]:
        """Return available data sources for the provided profile name."""
        profile_data_sources = []
        if profile in self.service_to_data_sources:

            for data_sources in self.service_to_data_sources[profile].values():

                profile_data_sources += data_sources
        
        return list(set(profile_data_sources))
    
    def update_incoming_history(self, time: float, instance_name: str, service: str) -> None:
        """Update incoming traffic history."""
        self.incoming_history[service][0].append(time)
        self.incoming_history[service][1].append(instance_name)
    
    def display_incoming_history(self, service: str, cluster_time: float=0.05) -> None:
        """Display the incoming traffic history."""
        if service not in self.incoming_history:
            raise ValueError(f"The service {service} isn't running on this machine.")

        times = self.incoming_history[service][0]
        sources = self.incoming_history[service][1]

        nb_cluster = int(times[-1] / cluster_time) + 1

        final_dict: Dict[str, List[int]] = {source: [0 for _ in range(nb_cluster)] for source in np.unique(sources)}
        end_time = cluster_time
        cluster = 0

        for t, s in zip(times, sources):

            if t <= end_time:

                final_dict[s][cluster] += 1
            
            else:

                end_time += cluster_time
                cluster += 1
                final_dict[s][cluster] += 1
        
        new_time = [(t + 0.5) * cluster_time for t in range(nb_cluster)]

        df = pd.DataFrame(final_dict)
        df['Time'] = new_time
        df = df.set_index('Time', drop=True)
        df.plot(figsize=(15, 4), title=f"Origin of traffic entering machine '{self.instance_name}' through the {service} port")
        plt.xlabel('Time')
        plt.ylabel('Connexion number')
        plt.legend()
        plt.show()
    
    def reset(self) -> None:
        """Reset the machine."""
        self.is_infected = deepcopy(self.__is_infected)
        self.incoming_history = {s: ([], []) for s in self.get_services()}


class Plug(Machine):
    """Connector for linking several machines."""

    def __init__(self, instance_name: str, platforms: List[str], connected_machines: List[str], outcomes: List[Outcome]=[], value: int=0, is_infected: bool=False, data_sources: Dict[str, List[Data_source]]=dict()):
        """Init the connected machine to the plug."""

        if len(connected_machines) <= 1:
            raise ValueError('A plug must at least link 2 machines instead of {}'.format(len(connected_machines)))

        name = 'Plug'
        url_image = 'Switch.jpg'
        super().__init__(name, instance_name, platforms, connected_machines, url_image, outcomes, value, is_infected, data_sources)


class Firewall(Machine):
    """Firewall class."""

    def __init__(self, incomings: List[Traffic], outgoings: List[Traffic], instance_name: str, platforms: List[str], connected_machines: List[str], outcomes: List[Outcome]=[], value: int=0, is_infected: bool=False, data_sources: Dict[str, List[Data_source]]=dict()):
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

    def __init__(self, instance_name: str, platforms: List[str], connected_machines: List[str], outcomes: List[Outcome]=[], value: int=0, is_infected: bool=False, data_sources: Dict[str, List[Data_source]]=dict()):
        """Init."""
        name = 'PC'
        url_image = 'PC.png'
        super().__init__(name, instance_name, platforms, connected_machines, url_image, outcomes, value, is_infected, data_sources)


class Server(Machine):
    """Server class."""

    def __init__(self, instance_name: str, platforms: List[str], connected_machines: List[str], outcomes: List[Outcome]=[], value: int=0, is_infected: bool=False, data_sources: Dict[str, List[Data_source]]=dict()):
        """Init."""
        name = 'Server'
        url_image = 'Server.png'
        super().__init__(name, instance_name, platforms, connected_machines, url_image, outcomes, value, is_infected, data_sources)


class Cloud(Machine):
    """Cloud class (external servers)."""

    def __init__(self, instance_name: str, platforms: List[str], connected_machines: List[str], outcomes: List[Outcome]=[], value: int=0, is_infected: bool=False, data_sources: Dict[str, List[Data_source]]=dict()):
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
