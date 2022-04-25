

from typing import Dict, List
import random

from .data import *
from .machine import Machine


class Activity:
    """Define an user activity during an environment step."""

    def __init__(self, source: str, activity: bool=False, where: str=None, action: str=None, service: str=None) -> None:
        """Init the activity.
        
        Input:
        source: The name of the machine from which the action originated (str)
        activity: indicates whether or not an activity is taking place (bool), default value is False
        where: the instance name of the machine where the activity takes place (str), default value is None
        action: the data source triggered (str), default value is None
        service: the service the user want to use to perform the activity (str), default value is None.
        Output: None
        """
        if not source:
            raise ValueError("Please indicate the source of the activity, even if the user have no activity.")

        self.activity = activity

        if activity:

            if not action:
                raise ValueError('Please indicate the source of the data requested.')
            
            if not where:
                raise ValueError('Please indicate where the action takes place.')
            
            if not service:
                raise ValueError('Please indicate the service the user want to use to perform the activity.')
            
        self.action = action
        self.where = where
        self.source = source
        self.service = service
    
    def is_activity(self) -> str:
        """Return whether there is activity or not."""
        return self.activity
    
    def get_where(self) -> str:
        """Return the instance name of the machine where the activity takes place."""
        return self.where

    def get_action(self) -> str:
        """Return the action comitted."""
        return self.action

    def get_source(self) -> str:
        """Return the action source."""
        return self.source

    def get_service(self) -> str:
        """Return the service used."""
        return self.service

class Profile:
    """Defines the user profile."""

    def __init__(self, name: str, data_source_distribution: Dict[Data_source, float]) -> None:
        """Init the profile.
        
        Input: 
        name: profile name (str)
        data_source_distribution: the probability distribution of triggering a data source (Dict[Data_source, float]).
        Output: None.
        """
        s = kahansum(np.array(list(data_source_distribution.values())))
        if s > 1 or s < 0:
            raise ValueError('The sum of the given probabilities : {}, is not between 0 and 1.'.format(s))
        
        if sum([1 for ds in data_source_distribution.keys() if isinstance(ds, UserAccount)]) < 1:
            raise ValueError("The profile isn't able to connect itself to machine. Please add a 'User account' data source.")

        self.name = name
        self.no_solicitation_prob = 1 - s
        self.data_source_distribution: Dict[Data_source, float] = dict()
        probs = np.cumsum([self.no_solicitation_prob] + list(data_source_distribution.values()))

        for ds, p in zip(data_source_distribution.keys(), probs[1:]):

            self.data_source_distribution[ds] = p
    
    def set_instance_name(self, instance_name: str) -> None:
        """Set instance name."""
        self.instance_name = instance_name

    def get_name(self) -> str:
        """Return the profile name."""
        return self.name
    
    def set_PC(self, pc_instance_name: str) -> None:
        """Set the pc the user is base on."""
        self.PC = pc_instance_name
    
    def get_PC_instance_name(self) -> str:
        """Return the PC instance name the user is based on."""
        if not self.PC:
            raise ValueError("The PC instance name the user is base on has not been yet set.")
        
        return self.PC

    def get_data_source_distribution(self) -> Dict[Data_source, float]:
        """Return the profile data source distribution."""
        return self.data_source_distribution
    
    def on_step(self, network_machines: List[Machine]) -> Activity:
        """Return an activity on an environment step with respect to the probability distribution.
        
        Input: network_machines (List[Machine]).
        Output: activity (Activity).
        """
        p = random.random()

        if p <= self.no_solicitation_prob:

            return Activity(source=self.PC)
        
        else:

            p -= self.no_solicitation_prob
            data_source_distribution = list(self.data_source_distribution.items())
            n_data_source = len(data_source_distribution)

            if p <= data_source_distribution[0][1]:

                data_source = data_source_distribution[0][0].get_data_source()
                machines = []
                
                for m in network_machines:
                    
                    can_perform, service = m.execute(self.get_name(), data_source)

                    if can_perform:

                        machines.append((m.get_instance_name(), service))

                if len(machines) == 0:

                    raise ValueError("The user {} wanted to use the data source {} but no machine in the environment provides this service.".format(self.name, data_source))
                
                else:

                    where, service = random.choice(machines)

                    return Activity(
                        activity=True,
                        action=data_source_distribution[0][0].call(),
                        where=where,
                        source=self.PC,
                        service=service
                        )

            for i in range(1, n_data_source):

                if p >= data_source_distribution[i-1][1] and p <= data_source_distribution[i][1]:

                    data_source = data_source_distribution[i][0].get_data_source()
                    machines = [(m.get_instance_name(), m.get_service_name(data_source)) for m in network_machines if m.is_data_source_available(data_source)]

                    if len(machines) == 0:

                        raise ValueError("The user {} wanted to use the data source {} but no machine in the environment provides this service.".format(self.name, data_source))                 
                    
                    else:

                        where, service = random.choice(machines)

                        return Activity(
                            activity=True,
                            action=data_source_distribution[i][0].call(),
                            where=where,
                            source=self.PC,
                            service=service
                            )
            
            if p >= data_source_distribution[n_data_source - 1][1]:

                data_source = data_source_distribution[n_data_source - 1][0].get_data_source()
                machines = [(m.get_instance_name(), m.get_service_name(data_source)) for m in network_machines if m.is_data_source_available(data_source)]

                if len(machines) == 0:

                    raise ValueError("The user {} wanted to use the data source {} but no machine in the environment provides this service.".format(self.name, data_source))                 
                
                else:

                    where, service = random.choice(machines)

                    return Activity(
                        activity=True,
                        action=data_source_distribution[n_data_source - 1][0].call(),
                        where=where,
                        source=self.PC,
                        service=service
                        )


class EnvironmentProfiles:
    """Defines all passiv actors in the environment."""

    def __init__(self, profiles: Dict[Profile, int], machines: List[Machine]) -> None:
        """Init profiles of passive environmental actors.
        
        Input: 
        profiles: dictionary associating to a given profile type its number of occurrences in the environment (Dict[Profile, int])
        machines: list of running machine isntance name (List[Machine]).
        """
        self.profiles_dict = profiles
        self.profiles: List[Profile] = []
        self.nb_profile = sum([n for _, n in profiles.items()])
        given_PC_count = 0
        self.network_machines = machines
        total_available_PC = [m.get_instance_name() for m in machines if (m.get_name() == 'PC' and not m.is_infected)]

        if len(total_available_PC) != self.nb_profile:
            raise ValueError('The environment does not have enough PCs for all profiles provided, number of profiles: {}, number of PCs: {}'.format(self.nb_profile, len(total_available_PC)))

        for profile, nb_profile in profiles.items():
            
            if not isinstance(profile, Profile):
                raise ValueError('Undefined profile class: {}'.format(profile))

            for j in range(nb_profile):

                p = profile.__class__()
                p.set_instance_name(profile.get_name() + '_' + str(j+1))
                p.set_PC(total_available_PC[given_PC_count])
                self.profiles.append(p)
                given_PC_count += 1
    
    def on_step(self) -> List[Activity]:
        """Return an array of activities.
        
        Output: Activities which is a list of length self.nb_passive_actors where each element corresponds to the activity of the profile associated with the index (ndarray[Activity]).
        """
        output = []

        for i in range(self.nb_profile):
            
            output.append(self.profiles[i].on_step(self.network_machines))
        
        return output

    def get_available_actions(self) -> List[str]:
        """Return the actions that each profile can perform."""
        res: List[str] = []

        for profile in self.profiles:

            data_sources = profile.get_data_source_distribution()

            for data_source in data_sources.keys():

                res += data_source.get_actions()
        
        return list(set(res))
    
    def get_profile_count(self) -> int:
        """Return the profile count."""
        return self.nb_profile
    
    def get_profiles(self) -> Dict[str, int]:
        """Return profiles."""
        return dict([(p.get_name(), n) for p, n in self.profiles_dict.items()])


class DSI(Profile):
    """DSI profile."""

    def __init__(self) -> None:
        name = 'DSI'
        data_source_distribution = {
            CloudStorage(): 0.2,
            LogonSession(): 0.2,
            CloudService(): 0.2,
            Driver(): 0.2,
            UserAccount(): 0.1
        }
        super().__init__(name, data_source_distribution)


class Dev(Profile):
    """Dev class."""

    def __init__(self) -> None:
        name = 'Dev'
        data_source_distribution = {
            Script(): 0.3,
            Process(): 0.2,
            File(): 0.1,
            UserAccount():0.1
        }
        super().__init__(name, data_source_distribution)