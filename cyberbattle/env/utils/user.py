

from typing import Dict, List, Tuple, Set
import random

from .data import *
from .machine import Machine, get_machines_by_name, trouble
from .network import Network
from .flow import Error
from ...utils.functions import kahansum
from ...utils.markov_models import MultiMarkovProcess


class Activity:
    """Define an user activity during an environment step."""

    def __init__(self, source: str=None, activity: bool=False, where: str=None, action: str=None, service: str=None, error: Error=None) -> None:
        """Init the activity.
        
        Input:
        source: The name of the machine from which the action originated (str), default value is None
        activity: indicates whether or not an activity is taking place (bool), default value is False
        where: the instance name of the machine where the activity takes place (str), default value is None
        action: the data source triggered (str), default value is None
        service: the service the user want to use to perform the activity (str), default value is None
        error: precises the error type (Error), for instance the activity can be stuck because of a firewall rule or a machine stoped running, default value is None.
        Output: None
        """
        self.activity = activity

        if activity:

            if not source:
                raise ValueError('Please indicate from where the action takes place.')

            if not action:
                raise ValueError('Please indicate the source of the data requested.')
            
            if not where:
                raise ValueError('Please indicate where the action takes place.')
            
            if not service:
                raise ValueError('Please indicate the service the user want to use to perform the activity.')

            if error is None:
                raise ValueError('Please indicate whether the activity was executed without any troubles or not.')
            
        self.action = action
        self.where = where
        self.source = source
        self.service = service
        self.error = error
    
    def get_description(self) -> str:
        """Describe the activity."""
        return f"Is activity : {self.activity}, source : {self.source}, target : {self.where}, port : {self.service}, data source triggered : {self.action}, is error : {self.error}"
    
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

    def get_error(self) -> Error:
        """Return the error type."""
        return self.error
    
    def is_error(self) -> bool:
        """Return whether the activity was executed without any troubles or not."""
        return self.error.value == Error.NO_ERROR.value


class Preferences:
    """Define the preferences of a user profile."""

    def __init__(self, source_local_prior: float=1, target_local_prior: float=1, source_prior: Dict[str, float]=None, target_prior: Dict[str, float]= None) -> None:
        """Init the user preferences.
        
        Input:
        source_local_prior: probability that the profile has to exercise an action from his associated PC (float)
        target_local_prior: probability that the profile has to exercise an action on his associated PC (float)
        source_prior: (optional) dictionary associating to a machine the probability that the source of the profile action is of its type (Dict[str, float])
        target_prior: (optional) dictionary associating to a machine the probability that the target of the profile action is of its type (Dict[str, float]).
        """
        if ( source_local_prior != 1 ) and (not source_prior ):
            raise ValueError(f"If the profile isn't performing only from his machine (found local_prior = {source_local_prior}), please indicate the source prior prior distribution.")

        if ( target_local_prior != 1 ) and (not target_prior ):
            raise ValueError(f"If the profile isn't performing only on his machine (found local_prior = {target_local_prior}), please indicate the target prior prior distribution.")

        if source_local_prior != 1:

            source_sum = kahansum(np.array(list(source_prior.values())))

            if source_sum != 1:
                raise ValueError(f"The sum over the source prior distribution is equal to {source_sum} but it must be equal to 1")

        if target_local_prior != 1:

            target_sum = kahansum(np.array(list(target_prior.values())))
            
            if target_sum != 1:
                raise ValueError(f"The sum over the source prior distribution is equal to {target_sum} but it must be equal to 1")

        self.source_local_prior = source_local_prior
        self.target_local_prior = target_local_prior
        self.source_prior = source_prior
        self.target_prior = target_prior

    def doing_action_from_its_PC(self) -> bool:
        """Return wheither the profile is performing its action from his PC or not."""
        p = random.random()
        return p <= self.source_local_prior

    def doing_action_on_its_PC(self) -> bool:
        """Return wheither the profile is performing its action on his PC or not."""
        p = random.random()
        return p <= self.target_local_prior
    
    def get_source_local_prior(self) -> float:
        """Return the source local prior probability."""
        return self.source_local_prior

    def get_target_local_prior(self) -> float:
        """Return the target local prior probability."""
        return self.target_local_prior
    
    def get_source_prior(self) -> Dict[str, float]:
        """Return the source prior distribution."""
        return self.source_prior

    def get_target_prior(self) -> Dict[str, float]:
        """Return the target prior distribution."""
        return self.target_prior
    
    def get_source_machine(self, user_pc: str) -> str:
        """Return the machine instance name from where the profile will perform his action."""
        if self.doing_action_from_its_PC():

            return user_pc

        return np.random.choice(list(self.source_prior.keys()), p=list(self.source_prior.values()))

    def get_target_machine(self, machines: List[Tuple[Machine, str]], user: str) -> Tuple[str, str]:
        """Return the machine instance name and service where the profile will perform among provided machines with respect to the target prior distribution."""
        target_machine_names = set(self.target_prior.keys())
        machines_name = set([m.get_instance_name() for (m, _) in machines])
        
        machines_kept = list(machines_name.intersection(target_machine_names))
        n = len(machines_kept)

        if n == 0:
            raise ValueError(f"The user {user} can't perform his action because machines providing the service in the environment : {machines_name} can't be used by him.")
        
        new_distribution = np.zeros(n)

        for i, m in enumerate(machines_kept):

            new_distribution[i] = self.target_prior[m]
        
        new_distribution /= np.sum(new_distribution)
        index = np.random.choice([i for i in range(n)], p=new_distribution)
        instance_name = machines_kept[index]

        return [(m.get_instance_name(), s) for (m, s) in machines if m.get_instance_name() == instance_name][0]


class Profile:
    """Defines the user profile."""

    def __init__(self, name: str, behavior: MultiMarkovProcess, preferences: Preferences) -> None:
        """Init the profile.
        
        Input: 
        name: profile name (str)
        behavior: a dynamic that defines the activity's user and so the data source he will trigger (MultiMarkovProcess)
        preferences: user activity location preference (Preferences).
        Output: None.
        """
        if sum([1 for data_source in behavior.get_markov_process_list() if not isinstance(data_source, Data_source)]) > 0:
            raise ValueError("Provided Markov process aren't Data_source instance.")

        self.name = name
        self.behavior = behavior        
        self.preferences = preferences
    
    def set_instance_name(self, instance_name: str) -> None:
        """Set instance name."""
        self.instance_name = instance_name
    
    def get_instance_name(self) -> str:
        """Return the user instance name."""
        return self.instance_name

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

    def check_policy(self, machines: List[Machine]) -> None:
        """Check if the profile can trigger all source data in the environment with the policy it has been assigned."""
        target_local_prior = self.preferences.get_target_local_prior()

        if target_local_prior == 1:

            user_PC = get_machines_by_name(self.PC, machines)[0]
            data_sources_in_PC = set(user_PC.get_available_datasources_profile(self.name))
            profile_data_source: Set[Data_source] = set([ds.get_data_source() for ds in self.behavior.get_markov_process_list()])

            if not profile_data_source.issubset(data_sources_in_PC):
                raise ValueError(f"Data sources the profil is able to trigger : {profile_data_source} aren't a subset of data sources on his associated PC : {data_sources_in_PC}")
    
    def on_step(self, network: Network) -> Activity:
        """Return an activity on an environment step with respect to the user's behavior and his preferences.
        
        Input: network (Network).
        Output: activity (Activity).
        """
        data_source: str = self.behavior.call()
        data_source_name = data_source.split(':')[0]

        if data_source_name == 'Quiet':

            return Activity()

        network_machines = network.get_machine_list()   
        from_where = self.preferences.get_source_machine(self.PC)
        user_PC = get_machines_by_name(self.PC, network_machines)[0]

        if self.preferences.doing_action_on_its_PC():

            performable, service = user_PC.execute(self.name, data_source_name)

            if performable:

                return Activity(
                    activity=True,
                    action=data_source,
                    where=self.PC,
                    source=from_where,
                    service=service,
                    error=Error.NO_ERROR
                    )
            
            else:
                
                return Activity()

        machines: List[Tuple[Machine, str]]= []
        
        for m in network_machines:
            
            can_perform, service = m.execute(self.get_name(), data_source_name)

            if can_perform:

                machines.append((m, service))

        if len(machines) == 0:
            raise ValueError(f"The user {self.instance_name} cannot trigger the data source {data_source_name} in the environment.")

        where, service = self.preferences.get_target_machine(machines, user=self.instance_name)

        path = network.get_path(from_where, where)
        
        if not isinstance(path, int):
            
            return Activity(
                activity=True,
                action=data_source,
                where=where,
                source=from_where,
                service=service,
                error=trouble(path, service)
                )

        return Activity(
            activity=True,
            action=data_source,
            where=where,
            source=from_where,
            service=service,
            error=Error.NO_ERROR
            )

    def reset(self) -> None:
        """Reset profile activity."""
        self.behavior.reset()


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
        total_available_PC = [m.get_instance_name() for m in machines if m.get_name() == 'PC']

        if len(total_available_PC) != self.nb_profile:
            raise ValueError('The environment does not have enough PCs for all profiles provided, number of profiles: {}, number of PCs: {}'.format(self.nb_profile, len(total_available_PC)))

        for profile, nb_profile in profiles.items():
            
            if not isinstance(profile, Profile):
                raise ValueError('Undefined profile class: {}'.format(profile))

            for j in range(nb_profile):

                p = profile.__class__(self.nb_profile + 1)
                p.set_instance_name(profile.get_name() + '_' + str(j+1))
                p.set_PC(total_available_PC[given_PC_count])
                p.check_policy(machines)
                self.profiles.append(p)
                given_PC_count += 1
    
    def on_step(self, network) -> List[Activity]:
        """Return an array of activities.
        
        Output: Activities which is a list of length self.nb_passive_actors where each element corresponds to the activity of the profile associated with the index (ndarray[Activity]).
        """
        output = []

        for i in range(self.nb_profile):
            
            output.append(self.profiles[i].on_step(network))
        
        return output

    def get_available_actions(self) -> List[str]:
        """Return the actions that each profile can perform."""
        res: List[str] = []

        for profile in self.profiles:

            data_sources: List[Data_source] = profile.behavior.get_markov_process_list()

            for data_source in data_sources:

                res += [ds.split(':')[0] for ds in data_source.get_actions()]
        
        return list(set(res))
    
    def get_profile_count(self) -> int:
        """Return the profile count."""
        return self.nb_profile
    
    def get_profiles(self) -> Dict[Profile, int]:
        """Return profiles."""
        return self.profiles
    
    def reset(self) -> None:
        """Reset profile activities."""
        for profile in self.profiles:

            profile.reset()
