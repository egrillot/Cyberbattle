

import json
import random
from enum import IntEnum
from typing import List, Dict, Tuple

from ..env.utils import flow, user
from ..env.utils.machine import Machine
from .scraping_mitre_attack import download_mitre_attacks


def low(l: List[str]) -> List[str]:
    """Return a list of string with only lower character."""
    return [s.lower() for s in l]

def get_available_attacks() -> List[Dict[str, List[str]]]:
    """Returns the mitre attack database as a dictionnary.
    
    Input: None
    Output: dict.
    """
    attacks_file = open('./vulnerabilities/recent_dl/enterprise-attack.json')
    attacks = json.load(attacks_file)['objects']
    attacks_file.close()

    return attacks


rights_link = {
    'Remote Desktop Users': flow.UserRight.LOCAL_USER,
    'User': flow.UserRight.LOCAL_USER,
    'Administrator': flow.UserRight.ADMIN,
    'SYSTEM': flow.UserRight.SYSTEM,
    'root': flow.UserRight.SYSTEM
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
        right = flow.UserRight.LOCAL_USER
        if 'x_mitre_permissions_required' in raw_dictionnary_attack:

            for permissions in raw_dictionnary_attack['x_mitre_permissions_required']:

                right = max(rights_link[permissions], right)
        
            self.requirements['right'] = right

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
    
    def get_outcomes(self) -> List[str]:
        """Return outcomes."""
        return self.outcomes
    
    def set_type(self, raw_dictionnary_attack: Dict[str, List[str]]) -> None:
        """Returns the attack type.
        
        Input: raw_dictionnary_attack (Dict[str, List[str]])
        Output: None.
        """
        self.type = None
        if 'x_mitre_remote_support' in raw_dictionnary_attack:
            if raw_dictionnary_attack['x_mitre_remote_support']:
                self.type = ActionType.REMOTE
            else:
                self.type = ActionType.LOCAL
        
        if not self.type:

            if 'x_mitre_permissions_required' in raw_dictionnary_attack:
                if max(rights_link[right] for right in raw_dictionnary_attack['x_mitre_permissions_required']) > flow.UserRight.LOCAL_USER:
                    self.type = ActionType.LOCAL
                else:
                    self.type = ActionType.REMOTE
    
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


def pass_filter(attack: Dict[str, List[str]], platforms_outcomes_couple_per_machine: Dict[str, Tuple[List[str], List[str]]]) -> Tuple[bool, int]:
    """Return whether the attack can be executed on a machine in the environment and if so which one.

    Input: 
    attack: mitre attack (Dict[str, List[str]])
    platforms_outcomes_couple_per_machine: dictionary that links machine instance name to its platforms and outcome phase names (Dict[str, Tuple[List[str], List[str]]]).
    Output: (Tuple[bool, int]).
    """
    instance_names = []

    if "kill_chain_phases" in attack:

        outcomes = [d["phase_name"].lower() for d in attack["kill_chain_phases"]]

        if "x_mitre_platforms" in attack: 

            platforms = set(low(attack['x_mitre_platforms']))
            match = False

            for instance_name, (p, o) in platforms_outcomes_couple_per_machine.items():

                if (len(set(p).intersection(platforms)) > 0) and (len(set(o).intersection(outcomes)) > 0):

                    match = True
                    break
            
            if match:
                
                if 'x_mitre_permissions_required' in attack:

                    if len(attack['x_mitre_permissions_required']):

                        if ("x_mitre_is_subtechnique" in attack and not attack["x_mitre_is_subtechnique"]) or ("x_mitre_is_subtechnique" not in attack):
                        
                            instance_names.append(instance_name)
    
    if len(instance_names) != 0:

        return True, instance_names

    return False, None


class AttackSet:
    """Defines the set of attack that the agent can perform in the environment."""

    def __init__(self, machines: List[Machine], profiles: user.EnvironmentProfiles, action_complexity: int=0, refresh=False) -> None:
        """Init.
        
        Input:
        machines: list of machines in the environment (List[Machine])
        profiles: passive actors in the environment (EnvironmentProfiles)
        action_complexity: Number of max allowed actions that cannot be performed by the profiles, default value: 0 (int)
        refresh: True if you want to update the local mitre attack database, defaut value: False (bool).
        """
        if refresh:
            download_mitre_attacks()

        mitre_attacks = get_available_attacks()
        random.shuffle(mitre_attacks)
        self.attacks_by_machines = {m.get_instance_name(): [] for m in machines}
        self.attacks = []
        self.id_to_attack = dict()
        platforms_outcomes_couple = dict()
        index = 0
        self.attack_count = 0
        self.data_sources = []
        self.available_actions = profiles.get_available_actions()
        not_performable_actions_count = 0
        
        for m in machines:

            outcomes = m.get_outcomes()
            
            if outcomes:
                
                platforms_outcomes_couple[m.get_instance_name()] = (low(m.get_platforms()), low([outcome.get_phase_name() for outcome in outcomes]))

        for mitre_attack in mitre_attacks:

            is_passing, instance_names = pass_filter(mitre_attack, platforms_outcomes_couple)

            if is_passing:

                attack = Attack(mitre_attack)
                data_sources = attack.get_data_sources()

                if len(data_sources) > 0:

                    if len(set(data_sources).intersection(set(self.available_actions))) == 0:

                        if not_performable_actions_count < action_complexity:

                            not_performable_actions_count += 1

                            attack.set_id(index)
                            attack.modify_data_sources([random.choice(data_sources)])
                            self.id_to_attack[index] = attack
                            for instance_name in instance_names:
                                self.attacks_by_machines[instance_name].append(attack)

                            self.attack_count += 1
                            self.attacks.append(attack)
                            self.data_sources += attack.get_data_sources()
                            index += 1
                    
                    else:

                            attack.set_id(index)
                            attack.modify_data_sources([random.choice(list(set(data_sources).intersection(set(self.available_actions))))])
                            self.id_to_attack[index] = attack
                            for instance_name in instance_names:
                                self.attacks_by_machines[instance_name].append(attack)

                            self.attack_count += 1
                            self.attacks.append(attack)
                            self.data_sources += attack.get_data_sources()
                            index += 1
        
        self.data_sources = list(set(self.data_sources))
        
    def get_attacks_by_machines(self) -> Dict[str, List[Attack]]:
        """Return the attack associated to each machines."""
        return self.attacks_by_machines
    
    def get_attack_count(self) -> int:
        """Return attack number that the attacker can use."""
        return self.attack_count

    def get_data_sources(self) -> List[str]:
        """Return the data source used with the attack set."""
        return self.data_sources

    def get_attacks(self) -> List[Attack]:
        """Return the attack list."""
        return self.attacks
