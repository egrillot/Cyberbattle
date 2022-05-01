"""Set up classes to propagate and manage informations through the network."""

from typing import Tuple
from enum import IntEnum
import numpy as np


class Error(IntEnum):
    """Determinate the error type during a user activity."""

    NO_ERROR = 0
    MACHINE_NOT_RUNNING = 1
    BLOCKED_BY_FIREWALL = 2
    MACHINE_NOT_PLUGED = 3

    def as_string(self) -> str:
        """Return the error type as a string."""
        if self.value == 0:

            return 'no error'
        
        elif self.value == 1:

            return "machine isn't running"
        
        elif self.value == 2:

            return 'blocked by a firewall'

        else:

            return "machine isn't pluged."

class Log:
    """Log class."""

    def __init__(
        self,
        source_id: int,
        target_id: int,
        action_id: int,
        service_id: int,
        error: int
    ) -> None:
        """Init the log.
        
        Input:
        source_id: source ip adress (int)
        target_id: machine ip adress sending the log (int)
        action_id: integer corresponding to the attempted data type on the target machine (int)
        service_id: service id corresponding to the service used by the user (int)
        error: 0 if the action was successful and otherwise it refers to the type of error occured (int).
        Output: None.
        """
        self.source_id = source_id
        self.target_id = target_id
        self.action_id = action_id
        self.service_id = service_id
        self.error = error

    def get_vector(self) -> np.ndarray:
        """Return the vector the SOC analyst agent will read.
        
        Output: array of shape (5,).
        """
        return np.array([self.source_id, self.target_id, self.action_id, self.service_id, self.error])


class Rule(IntEnum):
    """Rule applied to traffic"""

    ALLOWED = 0
    BLOCKED = 1


class Line(IntEnum):
    """Allow us to check whether a machine is efectively connected to a plug or not."""

    IN = 0
    OUT = 1


class Traffic:
    """Traffic class."""

    def __init__(
        self,
        port: str,
        rule: Rule
    ) -> None:
        """Define whether or not traffics through the provided port are allowed.
        
        Inputs: port name and a rule
        Output: None.
        """
        self.port = port
        self.rule = rule


class UserRight(IntEnum):
    """Define the user rights on a machine."""

    NO_ACCESS = 0
    LOCAL_USER = 1
    ADMIN = 2
    SYSTEM = 3
    MAXIMUM = 3


class Right:
    """Tracks user rights on a machine."""

    def __init__(
        self,
        right: UserRight=UserRight.NO_ACCESS
    ):
        """Inits the user right. 
        
        Input: right is by defaut a NO ACCESS right
        Output: None.
        """
        self.right = right
    
    def escalation(
        self,
        right: UserRight
    ):
        """Updates the user right."""
        self.right = max(self.right, right)
    
    def get_right(self):
        """Returns the current right."""
        return self.right


class Credential:
    """Credential class."""

    def __init__(self, port: str, machine: str, profile: str) -> None:
        """Init.
        
        Input:
        port: designates by which port the identifier is operational (str)
        machine: designates on which machine the identifier is operational (str)
        profile: profile usually using the password (str).
        """
        self.port = port
        self.machine = machine
        self.profile = profile
    
    def get_description(self) -> Tuple[str, str, str]:
        """Return the port name, machine name and credential."""
        return (self.port, self.machine, self.profile)
