

import random
import numpy as np

from typing import Dict
from ...env.utils.network import Network
from ...env.utils.user import EnvironmentProfiles, Activity
from ...env.utils.flow import Log

class SiemBox:
    """SiemBox class"""

    def __init__(self, profiles: EnvironmentProfiles, network: Network, action_to_id: Dict[str, int], profile_to_id: Dict[str, int], machine_to_id: Dict[str, int]) -> None:
        """Init the siem box."""
        self.profiles = profiles
        self.network = network
        self.profile_count = self.profiles.get_profile_count()
        self.action_to_id = action_to_id
        self.profile_to_id = profile_to_id
        self.machine_to_ip = machine_to_id
    
    def on_step(self, attacker_activity: Activity) -> np.ndarray:
        """Return what the different profiles did in the environment during the step."""
        res = np.zeros((self.profile_count + 1, 4), dtype=int)
        activities = self.profiles.on_step() + [attacker_activity]
        random.shuffle(activities)

        for activity in activities:

            if activity.is_activity():

                profile_name = activity.get_who()
                profile_id = self.profile_to_id(profile_name)
                machine = activity.get_where()
                machine_ip = self.machine_to_ip(machine)
                action = activity.get_action()
                action_id = self.action_to_id(action)

                

                log = Log(
                    source_id=profile_id,
                    target_id=machine_ip,
                    action_id=action_id,

                )


