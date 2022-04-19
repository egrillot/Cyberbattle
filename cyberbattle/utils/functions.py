
from ..env.utils.machine import Machine, Firewall
from typing import List, Tuple


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