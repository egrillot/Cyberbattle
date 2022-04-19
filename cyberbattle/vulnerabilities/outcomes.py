

from typing import List
from ..env.utils import flow

class Outcome:
    """Outcome class."""

    def __init__(self, phase_name: str, absolute_value: int=0) -> None:
        """Init."""
        self.phase_name = phase_name
        self.absolute_value = absolute_value
    
    def get_phase_name(self) -> str:
        """Return the phase name."""
        return self.phase_name

    def get_absolute_value(self) -> int:
        """Return the absolute reward."""
        return self.absolute_value


class LeakedCredentials(Outcome):
    """LeakedCredentials class."""

    def __init__(self, credentials: List[flow.Credential], absolute_value: int=0) -> None:
        """Init."""
        self.credentials = credentials
        phase_name = "credential-access"
        super().__init__(phase_name, absolute_value)
    
    def get(self) -> List[flow.Credential]:
        """Return credential list."""
        return self.credentials


class Escalation(Outcome):
    """Escalation class."""

    def __init__(self, userright: flow.UserRight, absolute_value: int=0) -> None:
        """Init."""
        self.userright = userright
        phase_name = "privilege-escalation"
        super().__init__(phase_name, absolute_value)

    def get(self) -> flow.UserRight:
        """Return the new user right."""
        return self.userright


class LeakedMachineIP(Outcome):
    """LeakedMachineIP class."""

    def __init__(self, machine_ip: List[int], absolute_value: int=0) -> None:
        """Init."""
        self.machine_ip = machine_ip
        phase_name = "discovery"
        super().__init__(phase_name, absolute_value)
    
    def get(self) -> List[int]:
        """Return discovered machines."""
        return self.machine_ip


class LateralMove(Outcome):
    """LateralMove class."""

    def __init__(self, absolute_value: int=0) -> None:
        """Init."""
        phase_name = "lateral-movement"
        super().__init__(phase_name, absolute_value)


class Reconnaissance(Outcome):
    """Reconnaissance class."""

    def __init__(self, data: str, absolute_value: int=0) -> None:
        """Init."""
        self.data = data
        phase_name = "reconnaissance"
        super().__init__(phase_name, absolute_value)
    
    def get(self) -> str:
        """Return the data."""
        return self.data