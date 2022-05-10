"""provide an environment with 3 internal servers and 1 external server (github projects) with as many as desired."""


import numpy as np
import random

from typing import List
from ...vulnerabilities.outcomes import LeakedCredentials, LeakedMachineIP, Collection
from ..utils.user import Profile, EnvironmentProfiles, Preferences
from ..utils.network import Network
from ..utils.flow import Traffic, Rule, UserRight, Credential
from ..utils.machine import Machine, Firewall, Client, Plug, Cloud, Server
from ..utils.data import *
from ...utils.markov_models import MultiMarkovProcess


client_services = {
    'Dev': {
        'sudo': ['Script', 'Process', 'File', 'User Account']
    },
    'DSI': {
        'sudo': ['Script', 'Process', 'File', 'User Account']
    }
}

googledrive_services = {
    'Dev': {
        'HTTPS': ['Driver', 'User Account']
    },
    'DSI': {
        'HTTPS': ['Logon Session', 'Driver', 'Cloud Storage', 'Cloud Service', 'User Account']
    }
}

servermail_services = {
    'Dev': {
        'HTTPS': ['User Account']
    },
    'DSI': {
        'HTTPS': ['Cloud Storage', 'Cloud Service', 'User Account']
    }
}

database_services = {
    'DSI': {
        'HTTPS': ['Cloud Storage', 'Cloud Service', 'User Account', 'Logon Session']
    }
}

def get_machine_list(num_client) -> List[Machine]:
    """Return list of machines to build the environment."""

    client_machines = [
        Client(instance_name='PC_{}'.format(i+1), platforms=['Windows'], connected_machines=['Switch_1'], value=0, data_sources=client_services)
        for i in range(num_client - 1)
        ] + [
            Client(instance_name='PC_{}'.format(num_client), platforms=['Windows'], connected_machines=['Switch_1'], value=0, data_sources=client_services,
            outcomes=[
                LeakedCredentials(credentials=[Credential(port='HTTPS', machine='MailServer', cred='MailDSI')])
                ])
        ]

    plug_machines = [
        Plug(instance_name='Switch_1', platforms=[], connected_machines=['PC_{}'.format(i+1) for i in range(num_client)]+['Router']),
        Plug(instance_name='Switch_2', platforms=[], connected_machines=['DatabaseServer', 'MailServer', 'CommunicationServer', 'Firewall_1']),
        Plug(instance_name='Router', platforms=[], connected_machines=['Switch_1', 'Firewall_1', 'Firewall_2'])
        ]
    
    internal_servers = [
        Server(instance_name='DatabaseServer', platforms=['Windows'], connected_machines=['Switch_2'], value=1000, data_sources=database_services,
        outcomes=[Collection(data='Confidential document', required_right=UserRight.LOCAL_USER, absolute_value=1000, flag=True)]), 
        Server(instance_name='MailServer', platforms=['IaaS'], connected_machines=['Switch_2'], value=200, data_sources=servermail_services,
        outcomes=[LeakedMachineIP(machine_ip=[
            'DatabaseServer', 'CommunicationServer', 'GoogleDrive'], required_right=UserRight.LOCAL_USER, flag=True)
            ]),
        Server(instance_name='CommunicationServer', platforms=['PRE'], connected_machines=['Switch_2'], value=200, data_sources=servermail_services) 
    ]

    external_servers = [
        Cloud(instance_name='GoogleDrive', platforms=['Google Workspace'], connected_machines=['Firewall_2'], value=500, data_sources=googledrive_services,
        outcomes=[LeakedCredentials(credentials=[Credential(port='HTTPS', machine='DatabaseServer', cred='AccessDSI')])])
    ]

    firewalls = [
        Firewall(
            instance_name='Firewall_1',
            platforms=[],
            connected_machines=['Router', 'Switch_2'],
            incomings=[
                Traffic(port='HTTPS', rule=Rule.ALLOWED),
                Traffic(port='sudo', rule=Rule.ALLOWED)
            ],
            outgoings=[
                Traffic(port='HTTPS', rule=Rule.ALLOWED),
                Traffic(port='sudo', rule=Rule.ALLOWED)
            ]
        ),
        Firewall(
            instance_name='Firewall_2',
            platforms=[],
            connected_machines=['Router', 'GoogleDrive'],
            incomings=[
                Traffic(port='HTTPS', rule=Rule.ALLOWED),
                Traffic(port='sudo', rule=Rule.ALLOWED)
            ],
            outgoings=[
                Traffic(port='HTTPS', rule=Rule.ALLOWED),
                Traffic(port='sudo', rule=Rule.ALLOWED)
            ]
        )
    ]

    machine_list: List[Machine] = client_machines + plug_machines + internal_servers + external_servers + firewalls

    for i, m in enumerate(machine_list):

        m.set_ip_adress(i)

    i = np.random.randint(0, num_client - 1)
    machine_list[i].infected_at_start()
    machine_list[i].outcomes = [LeakedMachineIP(machine_ip=['PC_{}'.format(i+1) for i in range(num_client)], required_right=UserRight.LOCAL_USER)]
    
    return machine_list


class DSI(Profile):
    """DSI profile."""

    def __init__(self, num_client: int) -> None:
        name = 'DSI'
        behavior = MultiMarkovProcess(
            markov_process_list=[
                CloudStorage(),
                LogonSession(),
                CloudService(),
                Driver(),
                UserAccount(),
                Quiet()
            ],
            markov_process_transition=np.array([
                [0.15, 0.15, 0.15, 0.15, 0.15, 0.25],
                [0.15, 0.15, 0.15, 0.15, 0.15, 0.25],
                [0.15, 0.15, 0.15, 0.15, 0.15, 0.25],
                [0.15, 0.15, 0.15, 0.15, 0.15, 0.25],
                [0.15, 0.15, 0.15, 0.15, 0.15, 0.25],
                [0.15, 0.15, 0.15, 0.15, 0.15, 0.25]
            ])
        )
        preferences = Preferences(
            source_local_prior = 0.8,
            target_local_prior = 0.3,
            source_prior = {
                'MailServer': 0.5,
                'GoogleDrive': 0.5
            },
            target_prior = {
                **{
                    'MailServer': 0.3,
                    'CommunicationServer': 0.2,
                    'GoogleDrive': 0.3,
            }, **dict([('PC_' + str(i+1), 0.2 / num_client) for i in range(num_client)])
            }
        )
        super().__init__(name, behavior, preferences)


class Dev(Profile):
    """Dev class."""

    def __init__(self, num_client: int) -> None:
        name = 'Dev'
        behavior = MultiMarkovProcess(
            markov_process_list=[
                Script(),
                Process(),
                File(),
                Driver(),
                UserAccount(),
                Quiet()
            ],
            markov_process_transition= np.array([
                [0.15, 0.15, 0.15, 0.15, 0.15, 0.25],
                [0.15, 0.15, 0.15, 0.15, 0.15, 0.25],
                [0.15, 0.15, 0.15, 0.15, 0.15, 0.25],
                [0.15, 0.15, 0.15, 0.15, 0.15, 0.25],
                [0.15, 0.15, 0.15, 0.15, 0.15, 0.25],
                [0.15, 0.15, 0.15, 0.15, 0.15, 0.25]
            ])
        )
        preferences = Preferences(
            source_local_prior=0.9,
            target_local_prior=0.9,
            source_prior={
                'GoogleDrive': 1
            },
            target_prior={
                **{
                    'GoogleDrive': 0.5,
                    'MailServer': 0.2,
                    'CommunicationServer': 0.2
                }, 
                **dict([('PC_' + str(i+1), 0.1 / num_client) for i in range(num_client)])
            }
        )
        super().__init__(name, behavior, preferences)


def get_little_environment_profiles(num_client) -> EnvironmentProfiles:
    """Return the environment profiles."""
    profiles = {
        Dev(num_client): num_client - 1,        
        DSI(num_client): 1
    }
    machine_list = get_machine_list(num_client)

    return EnvironmentProfiles(profiles, machine_list)
    

def get_little_environment_network(num_client) -> Network:
    """Return the network."""
    machine_list = get_machine_list(num_client)

    return Network(machine_list, name='Little_environment')
