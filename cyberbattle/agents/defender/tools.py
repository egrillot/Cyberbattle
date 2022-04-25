

import random
import numpy as np
import pandas as pd
import IPython.core.display as d
import matplotlib.pyplot as plt

from typing import Dict
from ...env.utils.network import Network
from ...env.utils.user import EnvironmentProfiles, Activity
from ...env.utils.flow import Log
from ...env.utils.machine import firewall_instances

class SiemBox:
    """SiemBox class"""

    def __init__(self, profiles: EnvironmentProfiles, network: Network, action_to_id: Dict[str, int], service_to_id: Dict[str, int]) -> None:
        """Init the siem box.
        
        Input:
        profiles: profiles operating in the environment (EnvironmentProfiles)
        network: the environment structure (Network)
        action_to_id: dictionnary where keys are actions and values are their id (Dict[str, int])
        service_to_id: dictionnary where keys are services and values are their id (Dict[str, int]).
        """
        self.profiles = profiles
        self.network = network
        self.profile_count = self.profiles.get_profile_count()
        self.action_to_id = action_to_id
        self.service_to_id = service_to_id
        self.instance_name_to_machine_ip = dict([(m.get_instance_name(), m) for m in network.get_machine_list()])
    
    def on_step(self, step_count: int, attacker_activity: Activity, display: bool=False) -> np.ndarray:
        """Return what the different profiles did in the environment during the step."""
        res = np.zeros((self.profile_count + 1, 5), dtype=int)
        activities = self.profiles.on_step() + [attacker_activity]

        to_display = []

        random.shuffle(activities)

        for i, activity in enumerate(activities):

            machine1 = activity.get_source()
            machine1_id = self.instance_name_to_machine_ip[machine1]

            if activity.is_activity():
                
                machine2 = activity.get_where()
                service = activity.get_service()
                machine2_id = self.instance_name_to_machine_ip[machine1]
                service_id = self.service_to_id[service]
                action = activity.get_action()
                action_id = self.action_to_id[action]

                if machine1 != machine2:

                    path = self.network.get_path(machine1, machine2)
                    firewalls = firewall_instances(path)
                    error = 0

                    for machine_before, firewall in firewalls:

                        if not firewall.is_passing(
                            port_name=service,
                            coming_from=machine_before
                        ):

                            error = 1
                            break

                    log = Log(
                        source_id=machine1_id,
                        target_id=machine2_id,
                        action_id=action_id,
                        service_id=service_id,
                        error=error
                    )

                    e = "yes, action blocked by a firewall" if error == 1 else "no"
                
                else:

                    log = Log(
                        source_id=machine1_id,
                        target_id=machine2_id,
                        action_id=action_id,
                        service_id=service_id,
                        error=0
                    )

                    e = "no"
                
                to_display.append(["yes", machine1, machine2, action, service, e])
            
            else:

                log = Log(
                    source_id=machine1_id,
                    target_id=-1,
                    action_id=-1,
                    service_id=-1,
                    error=-1
                )

                to_display.append(["no", "_", "_", "_", "_", "_"])

            res[i, :] = log.get_vector()
        
        if display:

            plt.title(f"Traffic during step {step_count}")
            d.display(pd.DataFrame(to_display, columns=["Activity", "Source ip adress", "Target ip adress", "Data source triggered", "Error"]))
            plt.show()

        return res
