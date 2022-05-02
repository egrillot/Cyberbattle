

from typing import List, Tuple
from ..env.utils.flow import Credential, Right, UserRight

class Outcome:
    """Outcome class."""

    def __init__(self, phase_name: str, absolute_value: int=0, required_right: Right=None, flag: bool=False) -> None:
        """Init."""
        self.phase_name = phase_name
        self.absolute_value = absolute_value
        self.required_right = required_right
        self.flag = flag
    
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

    def __init__(self, credentials: List[Credential], absolute_value: int=0, required_right: Right=None, flag: bool=False) -> None:
        """Init."""
        self.credentials = credentials
        phase_name = "credential-access"
        super().__init__(phase_name, absolute_value, required_right, flag)
    
    def get(self) -> Tuple[List[Credential], bool]:
        """Return credential list."""
        return self.credentials, self.flag


class Escalation(Outcome):
    """Escalation class."""

    def __init__(self, userright: UserRight, absolute_value: int=0, required_right: Right=None, flag: bool=False) -> None:
        """Init."""
        self.userright = userright
        phase_name = "privilege-escalation"
        super().__init__(phase_name, absolute_value, required_right, flag)

    def get(self) -> Tuple[UserRight, bool]:
        """Return the new user right."""
        return self.userright, self.flag


class LeakedMachineIP(Outcome):
    """LeakedMachineIP class."""

    def __init__(self, machine_ip: List[int], absolute_value: int=0, required_right: Right=None, flag: bool=False) -> None:
        """Init."""
        self.machine_ip = machine_ip
        phase_name = "discovery"
        super().__init__(phase_name, absolute_value, required_right, flag)
    
    def get(self) -> Tuple[List[int], bool]:
        """Return discovered machines."""
        return self.machine_ip, self.flag


class LateralMove(Outcome):
    """LateralMove class."""

    def __init__(self, absolute_value: int=0, required_right: Right=None, flag: bool=False) -> None:
        """Init."""
        phase_name = "lateral-movement"
        super().__init__(phase_name, absolute_value, required_right, flag)
    
    def get(self) -> Tuple[str, bool]:
        """Return nothing."""
        return self.phase_name, self.flag


class Reconnaissance(Outcome):
    """Reconnaissance class."""

    def __init__(self, data: str, absolute_value: int=0, required_right: Right=None, flag: bool=False) -> None:
        """Init."""
        self.data = data
        phase_name = "reconnaissance"
        super().__init__(phase_name, absolute_value, required_right, flag)
    
    def get(self) -> Tuple[str, bool]:
        """Return the data."""
        return self.data, self.flag

class Collection(Outcome):
    """Collection class."""

    def __init__(self, data: str, absolute_value: int=0, required_right: Right=None, flag: bool=False) -> None:
        """Init."""
        self.data = data
        phase_name = "collection"
        super().__init__(phase_name, absolute_value, required_right, flag)
    
    def get(self) -> Tuple[str, bool]:
        """Return the data."""
        return self.data, self.flag
