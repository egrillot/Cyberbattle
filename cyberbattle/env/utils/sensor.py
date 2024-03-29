

import random
import numpy as np
import pandas as pd
import IPython.core.display as d
import matplotlib.pyplot as plt
import time

from typing import Dict, List
from ...env.utils.network import Network
from ...env.utils.user import EnvironmentProfiles, Activity
from ...env.utils.flow import Log

class SiemBox:
    """SiemBox class"""

    def __init__(self, profiles: EnvironmentProfiles, network: Network, action_to_id: Dict[str, int], service_to_id: Dict[str, int], start_time: float) -> None:
        """Init the siem box.
        
        Input:
        profiles: profiles operating in the environment (EnvironmentProfiles)
        network: the environment structure (Network)
        action_to_id: dictionnary where keys are actions and values are their id (Dict[str, int])
        service_to_id: dictionnary where keys are services and values are their id (Dict[str, int])
        start_time: starting time of the simulation (float).
        """
        self.profiles = profiles
        self.network = network
        self.profile_count = self.profiles.get_profile_count()
        self.action_to_id = action_to_id
        self.service_to_id = service_to_id
        self.instance_name_to_machine = dict([(m.get_instance_name(), m) for m in network.get_machine_list()])
        self.instance_name_to_machine_ip = dict([(m.get_instance_name(), m.get_ip_adress()) for m in network.get_machine_list()])
        self.history: Dict[float, List[str]] = dict()
        self.__start_time = start_time
    
    def on_step(self, network: Network, step_count: int, attacker_activity: Activity, start_time: float, display: bool=False) -> np.ndarray:
        """Return what the different profiles did in the environment during the step."""
        res = np.zeros((self.profile_count + 1, 5), dtype=int)
        activities = self.profiles.on_step(network) + [attacker_activity]

        to_display = []

        random.shuffle(activities)

        for i, activity in enumerate(activities):

            t = time.time() - self.__start_time

            if activity.is_activity():
                
                machine1 = activity.get_source()
                machine1_id = self.instance_name_to_machine_ip[machine1]

                action = activity.get_action()
                machine2 = activity.get_where()
                service = activity.get_service()                
                self.instance_name_to_machine[machine2].update_incoming_history(time.time() - start_time, machine1, service)

                machine2_id = self.instance_name_to_machine_ip[machine1]
                service_id = self.service_to_id[service]

                action = activity.get_action()
                action_id = self.action_to_id[action]

                error = activity.get_error()
                error_type = error.value
                e = error.as_string()

                log = Log(
                    time= t,
                    source_id=machine1_id,
                    target_id=machine2_id,
                    action_id=action_id,
                    service_id=service_id,
                    error=error_type
                )

                self.history
        
                to_display.append(["yes", machine1, machine2, action, service, e])    

            else:

                log = Log(
                    time=None,
                    source_id=-1,
                    target_id=-1,
                    action_id=-1,
                    service_id=-1,
                    error=-1
                )

                to_display.append(["no", "_", "_", "_", "_", "_"])

            res[i, :] = log.get_vector()
            self.history[t] = to_display
        
        if display:

            print('\n')
            print(f"Traffic during step {step_count}")
            df = pd.DataFrame(to_display, columns=["Activity", "Source ip adress", "Target ip adress", "Data source triggered", "Port", "Error"])
            d.display(df)
            plt.show()

        return res
    
    def get_history(self) -> List[List[str]]:
        """Return the traffic history seen wihtin the siem box."""
        return self.history

    def reset(self, new_start_time: float) -> None:
        """Reset the siem box."""
        self.history = dict()
        self.__start_time = new_start_time
