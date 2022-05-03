

import json
import random
import numpy as np
#from pyattck import Attck
from importlib.resources import open_binary
from enum import IntEnum
from typing import List, Dict, Tuple

from ..env.utils.flow import UserRight
from ..env.utils.user import EnvironmentProfiles
from ..env.utils.machine import Machine
from . import recent_dl

def low(l: List[str]) -> List[str]:
    """Return a list of string with only lower character."""
    return [s.lower() for s in l]


#def download_mitre_enterprise_attack():
#    """Download the last version of enterprise attack."""
#    attack = Attck()
#    for technique in attack.enterprise.techniques:
#        print(technique.id)
#        print(technique.name)

def get_available_attacks() -> List[Dict[str, List[str]]]:
    """Returns the mitre attack database as a dictionnary.
    
    Input: None
    Output: dict.
    """
    with open_binary(recent_dl, "enterprise-attack.json") as attacks_file:
        attacks = json.load(attacks_file)
    
    return attacks["objects"]


rights_link = {
    'Remote Desktop Users': UserRight.LOCAL_USER,
    'User': UserRight.LOCAL_USER,
    'Administrator': UserRight.ADMIN,
    'SYSTEM': UserRight.SYSTEM,
    'root': UserRight.SYSTEM
}


class ActionType(IntEnum):
    """Defines how to operate the action."""

    LOCAL = 0
    REMOTE = 1
    CONNECT = 2


class Attack:
    """Attack class."""

    def __init__(self, raw_dictionnary_attack: Dict[str, List[str]]):
        """Inits the attack requirements, type, outcome and the requested data sources"""
        self.set_requirements(raw_dictionnary_attack)
        self.set_data_sources(raw_dictionnary_attack)
        self.set_outcomes(raw_dictionnary_attack)
        self.set_type(raw_dictionnary_attack)
        self.set_name(raw_dictionnary_attack)
        self.set_url(raw_dictionnary_attack)
        self.set_description(raw_dictionnary_attack)
    
    def set_requirements(self, raw_dictionnary_attack: Dict[str, List[str]]) -> None:
        """Returns requirements to execute the attack.
        
        Input: raw_dictionnary_attack (Dict[str, List[str]])
        Output: None.
        """
        self.requirements = dict()
        if 'x_mitre_permissions_required' in raw_dictionnary_attack:

            self.requirements['right'] = min([rights_link[permission] for permission in raw_dictionnary_attack['x_mitre_permissions_required']])

        else:

            self.requirements['right'] = UserRight.NO_ACCESS

        if 'x_mitre_platforms' in raw_dictionnary_attack:
            
            self.requirements['platforms']  = raw_dictionnary_attack['x_mitre_platforms']
    
    def get_requirements(self) -> Dict[str, List[str]]:
        """Return the requirements to perform the attack."""
        return self.requirements
    
    def set_data_sources(self, raw_dictionnary_attack: Dict[str, List[str]]) -> None:
        """Returns requesting data sources if the attacker performs the attack.
        
        Input: raw_dictionnary_attack (Dict[str, List[str]])
        Output: None.
        """
        self.data_sources = []
        if 'x_mitre_data_sources' in raw_dictionnary_attack:

            self.data_sources = raw_dictionnary_attack['x_mitre_data_sources']
    
    def modify_data_sources(self, new_data_sources: List[str]) -> None:
        """Modify data sources used."""
        self.data_sources = new_data_sources
    
    def get_data_sources(self) -> List[str]:
        """Return requested data sources."""
        return self.data_sources
    
    def set_outcomes(self, raw_dictionnary_attack: Dict[str, List[str]]) -> None:
        """Returns outcomes performing the attack.
        
        Input: raw_dictionnary_attack (Dict[str, List[str]])
        Output: None.
        """
        self.outcomes = []
        if 'kill_chain_phases' in raw_dictionnary_attack:
            
            for outcome in raw_dictionnary_attack['kill_chain_phases']:
                self.outcomes.append(outcome['phase_name'])
    
    def modify_outcomes(self, new_outcome: List[str]) -> None:
        """Modify by a unique outcome."""
        self.outcomes = new_outcome
    
    def get_outcomes(self) -> List[str]:
        """Return outcomes."""
        return self.outcomes
    
    def set_type(self, raw_dictionnary_attack: Dict[str, List[str]]) -> None:
        """Returns the attack type.
        
        Input: raw_dictionnary_attack (Dict[str, List[str]])
        Output: None.
        """
        self.type = None
        if 'x_mitre_permissions_required' in raw_dictionnary_attack:
            if max(rights_link[right] for right in raw_dictionnary_attack['x_mitre_permissions_required']) > UserRight.LOCAL_USER:
                self.type = ActionType.LOCAL
            else:
                self.type = ActionType.REMOTE
        
        else:
            self.type = ActionType.REMOTE
    
        if 'x_mitre_remote_support' in raw_dictionnary_attack:
            if raw_dictionnary_attack['x_mitre_remote_support']:
                self.type = ActionType.REMOTE
            else:
                self.type = ActionType.LOCAL
 
        if not self.type:
       
            self.type = ActionType.LOCAL
    
    def get_type(self) -> ActionType:
        """Return action type."""
        return self.type
    
    def set_name(self, raw_dictionnary_attack: Dict[str, List[str]]) -> None:
        """Returns the attack name.
        
        Input: raw_dictionnary_attack (Dict[str, List[str]])
        Output: name (str).
        """
        self.name = None
        if 'name' in raw_dictionnary_attack:
            self.name = raw_dictionnary_attack['name']
    
    def get_name(self) -> str:
        """Return the attack name."""
        return self.name
    
    def set_url(self, raw_dictionnary_attack: Dict[str, List[str]]) -> None:
        """Returns the attack name.
        
        Input: raw_dictionnary_attack (Dict[str, List[str]])
        Output: None.
        """
        self.url = None
        if 'url' in raw_dictionnary_attack:
            self.url = raw_dictionnary_attack['url']

    def get_url(self) -> str:
        """Return the attack url."""
        return self.url

    def set_description(self, raw_dictionnary_attack: Dict[str, List[str]]) -> None:
        """Returns the attack name.
        
        Input: raw_dictionnary_attack (dict)
        Output: description (str).
        """
        self.description = None
        if 'description' in raw_dictionnary_attack:
            self.description = raw_dictionnary_attack['description']

    def get_description(self) -> str:
        """Return the attack description."""
        return self.description
    
    def set_id(self, id: int) -> None:
        """Set the attack id."""
        self.id = id
    
    def get_id(self) -> int:
        """Return the attack id."""
        return self.id


def pass_filter(
    attack: Dict[str, List[str]],
    platforms_outcomes_couple_per_machine: Dict[str, Tuple[List[str], List[str], List[Tuple[str, UserRight]]]]
    ) -> Tuple[bool, List[str], Dict[str, List[Tuple[bool, str, UserRight]]], List[str]]:
    """Return whether the attack can be executed on a machine in the environment and if so which one.

    Input: 
    attack: mitre attack (Dict[str, List[str]])
    platforms_outcomes_couple_per_machine: dictionary that links machine instance name to its platforms, available data sources that can be triggered and outcome phase names (Dict[str, Tuple[List[str], List[str], List[Tuple[str, UserRight]]]]).
    Output: a tuple refering to whether the attack successfully passed the filter, the machine instance name the attack can be applied, the available outcomes in each machine and a list of data source that can be triggered by the targetable machines (Tuple[bool, List[str], Dict[str, List[Tuple[bool, str, UserRight]]], List[str]]).
    """
    instance_names = []
    matched_outcome_count: Dict[str, List[bool]] = dict()
    data_sources_intersection = set()

    if "name" in attack:

        if "x_mitre_data_sources" in attack:

            if "kill_chain_phases" in attack:

                outcomes = [d["phase_name"].lower() for d in attack["kill_chain_phases"]]

                if "x_mitre_platforms" in attack: 

                    platforms = set(low(attack['x_mitre_platforms']))

                    for instance_name, (p, data_sources, phase_names_and_userrights) in platforms_outcomes_couple_per_machine.items():

                        if len(set(p).intersection(platforms)) > 0:

                            data_sources_intersection_on_machine = set([ds.split(':')[0] for ds in attack['x_mitre_data_sources']]).intersection(set(data_sources))

                            if len(data_sources_intersection_on_machine) > 0:

                                data_sources_intersection.update(data_sources_intersection_on_machine)
                                matched_outcome_count[instance_name] = [(False, '') * len(phase_names_and_userrights)]

                                for i, (phase_name, userright) in enumerate(phase_names_and_userrights):
        
                                    if phase_name in outcomes:

                                        instance_names.append(instance_name)
                                        matched_outcome_count[instance_name][i] = (True, phase_name, userright)

    if len(instance_names) != 0:

        return True, instance_names, matched_outcome_count, list(data_sources_intersection)

    return False, None, None, None


class AttackSet:
    """Defines the set of attack that the agent can perform in the environment."""

    def __init__(self, machines: List[Machine], profiles: EnvironmentProfiles, max_per_outcomes: int=1, use_static_version=True) -> None:
        """Init.
        
        Input:
        machines: list of machines in the environment (List[Machine])
        profiles: passive actors in the environment (EnvironmentProfiles)
        max_per_outcomes: indicates how many attacks are retained for each outcome per machine at most, default value: 1 (int)
        use_static_version: indicates wheither the attacks are scraped on the local data or scraped using pyattck
        refresh: True if you want to update the local mitre attack database, defaut value: False (bool).
        """
        #download_mitre_enterprise_attack()
        if use_static_version:
            mitre_attacks = get_available_attacks()
        random.shuffle(mitre_attacks)
        attacks_by_machines: Dict[str, List[List[Attack]]] = {m.get_instance_name(): [[] * len(m.get_outcomes())] for m in machines}
        self.all_attacks = []
        self.all_attacks_names = []
        self.attacks: Dict[str, List[Attack]] = {m.get_instance_name(): [] for m in machines}
        platforms_outcomes_couple = dict()
        self.attack_count = 0
        self.data_sources: List[str] = []
        self.available_actions = profiles.get_available_actions()

        for m in machines:

            outcomes = m.get_outcomes()
            
            if outcomes:
                
                platforms_outcomes_couple[
                    m.get_instance_name()
                    ] = (low(m.get_platforms()), m.get_flat_data_sources(),
                     [(outcome.get_phase_name().lower(), outcome.get_required_right()) for outcome in outcomes]
                     )

        for mitre_attack in mitre_attacks:

            is_passing, instance_names, matched_outcome_count, data_sources_intersection = pass_filter(mitre_attack, platforms_outcomes_couple)

            if is_passing:

                attack = Attack(mitre_attack)
                attack.modify_data_sources(data_sources_intersection)
                    
                for instance_name in instance_names:

                    for i, match in enumerate(matched_outcome_count[instance_name]):
                        
                        if match[0]:

                            required_right = match[2]
                            attack_kept = False
                            attack_type = attack.get_type()

                            if required_right is not None:

                                if attack_type == ActionType.LOCAL:

                                    attack_kept = True

                            else:
                                
                                if attack_type == ActionType.REMOTE:

                                    attack_kept = True

                            if attack_kept:
                                
                                attack.modify_outcomes(new_outcome=[match[1]])
                                attacks_by_machines[instance_name][i].append(attack)
        
        for instance_name, attacks_per_outcomes in attacks_by_machines.items():

            for attacks in attacks_per_outcomes:

                if attacks:

                    attacks_remained: List[Attack] = np.random.choice(attacks, size=max_per_outcomes).tolist() if len(attacks) > max_per_outcomes else attacks
                    self.attacks[instance_name] += attacks_remained

                    for attack in attacks_remained:

                        if attack.get_name() not in self.all_attacks_names:

                            self.all_attacks.append(attack)
                            self.all_attacks_names.append(attack.get_name())
                            self.data_sources += attack.get_data_sources()
                            attack.set_id(self.attack_count)
                            self.attack_count += 1
        
        self.data_sources = list(set(self.data_sources))
        
    def get_attacks_by_machines(self) -> Dict[str, List[Attack]]:
        """Return the attack associated to each machines."""
        return self.attacks
    
    def get_attack_count(self) -> int:
        """Return attack number that the attacker can use."""
        return self.attack_count

    def get_data_sources(self) -> List[str]:
        """Return the data source used within the attack set."""
        return set([ds.split(':')[0] for ds in self.data_sources])

    def get_attacks(self) -> List[Attack]:
        """Return the attack list."""
        return self.all_attacks

    def get_attacks_by_machines_string(self) -> Dict[str, Dict[str, Dict[str, str]]]:
        """Return the attack description associated to each machines."""
        return {machine: {attack.get_name(): {
            "data sources triggered": attack.get_data_sources()[0], 
            "phase name": attack.get_outcomes()[0],
            "Type": "Remote" if attack.get_type() == ActionType.REMOTE else "Local"
         } for attack in attacks} for machine, attacks in self.attacks.items()}
