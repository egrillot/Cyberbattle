

from typing import List
from ..env.utils.flow import Credential, Right, UserRight

class Outcome:
    """Outcome class."""

    def __init__(self, phase_name: str, absolute_value: int=0, required_right: Right=None) -> None:
        """Init."""
        self.phase_name = phase_name
        self.absolute_value = absolute_value
        self.required_right = required_right
    
    def get_phase_name(self) -> str:
        """Return the phase name."""
        return self.phase_name

    def get_absolute_value(self) -> int:
        """Return the absolute reward."""
        return self.absolute_value

    def get_required_right(self) -> Right:
        """Return the required right."""
        return self.required_right
    
    def get(self) -> None:
        raise NotImplementedError


class LeakedCredentials(Outcome):
    """LeakedCredentials class."""

    def __init__(self, credentials: List[Credential], absolute_value: int=0, required_right: Right=None) -> None:
        """Init."""
        self.credentials = credentials
        phase_name = "credential-access"
        super().__init__(phase_name, absolute_value, required_right)
    
    def get(self) -> List[Credential]:
        """Return credential list."""
        return [cred.get_description() for cred in self.credentials]


class Escalation(Outcome):
    """Escalation class."""

    def __init__(self, userright: UserRight, absolute_value: int=0, required_right: Right=None) -> None:
        """Init."""
        self.userright = userright
        phase_name = "privilege-escalation"
        super().__init__(phase_name, absolute_value, required_right)

    def get(self) -> UserRight:
        """Return the new user right."""
        return self.userright


class LeakedMachineIP(Outcome):
    """LeakedMachineIP class."""

    def __init__(self, machine_ip: List[int], absolute_value: int=0, required_right: Right=None) -> None:
        """Init."""
        self.machine_ip = machine_ip
        phase_name = "discovery"
        super().__init__(phase_name, absolute_value, required_right)
    
    def get(self) -> List[int]:
        """Return discovered machines."""
        return self.machine_ip


class LateralMove(Outcome):
    """LateralMove class."""

    def __init__(self, absolute_value: int=0, required_right: Right=None) -> None:
        """Init."""
        phase_name = "lateral-movement"
        super().__init__(phase_name, absolute_value, required_right)
    
    def get(self) -> str:
        """Return nothing."""
        return self.phase_name


class Reconnaissance(Outcome):
    """Reconnaissance class."""

    def __init__(self, data: str, absolute_value: int=0, required_right: Right=None) -> None:
        """Init."""
        self.data = data
        phase_name = "reconnaissance"
        super().__init__(phase_name, absolute_value, required_right)
    
    def get(self) -> str:
        """Return the data."""
        return self.data

class Collection(Outcome):
    """Collection class."""

    def __init__(self, data: str, absolute_value: int=0, required_right: Right=None) -> None:
        """Init."""
        self.data = data
        phase_name = "collection"
        super().__init__(phase_name, absolute_value, required_right)
    
    def get(self) -> str:
        """Return the data."""
        return self.data
