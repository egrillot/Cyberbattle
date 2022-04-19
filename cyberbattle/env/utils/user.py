

from typing import Dict, List
import random
import numpy as np

from .data import *
from .machine import Machine, get_machines_by_name


class Activity:
    """Define an user activity during an environment step."""

    def __init__(self, activity: bool, where: str, who: str, action: str=None) -> None:
        """Init the activity.
        
        Input:
        activity: if there is one (bool)
        where: the instance name of the machine where the activity takes place (str)
        action: the action comitted (str), default value is None
        who: the profile instance name who is doing the action.
        Output: None
        """
        self.activity = activity

        if activity:

            if not action:
                raise ValueError('Please indicate the source of the data requested.')
            
        self.action = action
        self.where = where
        self.who = who
    
    def is_activity(self) -> str:
        """Return whether there is activity or not."""
        return self.activity
    
    def get_where(self) -> str:
        """Return the instance name of the machine where the activity takes place."""
        return self.where

    def get_action(self) -> str:
        """Return the action comitted."""
        return self.action

    def get_who(self) -> str:
        """Return who did the action."""
        return self.who


class Profile:
    """Defines the user profile."""

    def __init__(self, name: str, data_source_distribution: Dict[Data_source, float], based_on: List[str]) -> None:
        """Init the profile.
        
        Input: 
        name: profile name (str)
        data_source_distribution: the probability distribution of requesting a data source (Dict[Data_source, float])
        based_on: typical machine instance names commonly used by the profile (List[str]).
        Output: None.
        """
        s = sum(data_source_distribution.values())
        if s > 1 or s < 0:
            raise ValueError('The sum of the given probabilities : {}, is not between 0 and 1.'.format(s))

        if 'PC' not in based_on:
            print('One PC has been added to the list of machines used by the profile: {}'.format(name))
            based_on.append('PC')
        
        no_solicitation_prob = 1 - s
        
        if len(based_on) <= 1:
            raise ValueError('Please indicate at least one more machine ip adress where the profile may have an activity')

        self.name = name
        self.data_source_distribution = data_source_distribution
        self.no_solicitation_prob = no_solicitation_prob
        self.based_on = based_on
    
    def set_instance_name(self, instance_name: str) -> None:
        """Set instance name."""
        self.instance_name = instance_name

    def get_name(self) -> str:
        """Return the profile name."""
        return self.name
    
    def get_machines_based_on(self) -> List[str]:
        """Return machines where the profile can operate."""
        return self.based_on
    
    def get_data_source_distribution(self) -> Dict[Data_source, float]:
        """Return the profile data source distribution."""
        return self.data_source_distribution
    
    def on_step(self, p: float) -> Activity:
        """Return an activity on an environment step with respect to the probability distribution.
        
        Input: p which correspond to a probability following an uniform law over [0, 1] (float).
        Output: activity (Activity).
        """
        where = random.choice(self.based_on)

        if p <= self.no_solicitation_prob:

            return Activity(
                activity=False,
                where=where,
                who=self.instance_name
                )
        
        else:

            p -= self.no_solicitation_prob
            data_source_distribution = list(self.data_source_distribution.items())
            n_data_source = len(data_source_distribution)

            if p <= data_source_distribution[0][1]:

                return Activity(
                    activity=True,
                    action=data_source_distribution[0][0].call(),
                    where=where,
                    who=self.instance_name
                    )

            for i in range(1, n_data_source):

                if p >= data_source_distribution[i-1][1] and p <= data_source_distribution[i][1]:

                    return Activity(
                        activity=True,
                        action=data_source_distribution[i][0].call(),
                        where=where,
                        who=self.instance_name
                        )


class DSI(Profile):
    """DSI profile."""

    def __init__(self, based_on: List[str]) -> None:
        name = 'DSI'
        data_source_distribution = {
            CloudStorage(): 0.4,
            CloudService(): 0.4,
            ScheduledJob(): 0.1
        }
        super().__init__(name, data_source_distribution, based_on)


class Dev(Profile):
    """Dev class."""

    def __init__(self, based_on: List[str]) -> None:
        name = 'Dev'
        data_source_distribution = {
            Script(): 0.4,
            Process(): 0.4
        }
        super().__init__(name, data_source_distribution, based_on)


class EnvironmentProfiles:
    """Defines all passiv actors in the environment."""

    def __init__(self, profiles: Dict[(Profile, int)], machines: List[Machine]) -> None:
        """Init profiles of passive environmental actors.
        
        Input: 
        profiles: dictionary associating to a given profile type its number of occurrences in the environment (Dict[(str, int)])
        machines: list of running machine isntance name (List[Machine]).
        """
        self.profiles: List[Profile] = []
        self.nb_profile = sum([n for (p, n) in profiles.items() if len(p.get_machines_based_on()) != 0])
        given_PC_count = 0
        total_available_PC = [m.get_instance_name() for m in machines if (m.get_name() == 'PC' and not m.is_infected)]

        if len(total_available_PC) != self.nb_profile:
            raise ValueError('The environment does not have enough PCs for all profiles provided, number of profiles: {}, number of PCs: {}'.format(self.nb_profile, len(total_available_PC)))

        for profile in profiles.keys():
            
            if not isinstance(profile, Profile):
                raise ValueError('Undefined profile class: {}'.format(profile))

            nb_profile = profiles[profile]
            machines_based_on = profile.get_machines_based_on()
            potential_machines : List[Machine] = []

            for machine in machines_based_on:

                remained_machines = get_machines_by_name(machine, machines)

                if len(set(total_available_PC).intersection(remained_machines)) != 0:

                    remained_machines = [total_available_PC[given_PC_count]]
                    given_PC_count += 1
                
                potential_machines += remained_machines

            for j in range(nb_profile):

                p = profile.__class__(based_on=[m.get_instance_name() for m in potential_machines])
                p.set_instance_name(profile.get_name() + '_' + str(j+1))
                self.profiles.append(p)
    
    def on_step(self) -> List[Activity]:
        """Return an array of activities.
        
        Output: Activities which is an array of shape (self.nb_passive_actors,) where each element corresponds to the activity of the profile associated with the index (ndarray[Activity]).
        """
        output = []

        p = random.random()

        for i in range(self.nb_profile):

            output.append(self.profiles[i].on_step(p))
        
        return output

    def get_available_actions(self) -> List[str]:
        """Return the actions that each profile can perform."""
        res: List[str] = []

        for profile in self.profiles:

            data_sources = profile.get_data_source_distribution()

            for data_source in data_sources.keys():

                res += [f'{data_source.get_data_source()}: {a}' for a in data_source.get_actions()]
        
        return list(set(res))
    
    def get_profile_count(self) -> int:
        """Return the profile count."""
        return self.nb_profile