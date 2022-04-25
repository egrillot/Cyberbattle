"""provide an environment with 3 internal servers and 1 external server (github projects) with as many as desired."""


from typing import List
from ...vulnerabilities.outcomes import LeakedCredentials, LeakedMachineIP, Collection
from ..utils.user import DSI, Dev, EnvironmentProfiles
from ..utils.network import Network
from ..utils.flow import Traffic, Rule, UserRight, Credential
from ..utils.machine import Machine, Firewall, Client, Plug, Cloud, Server


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
    'DSI': {
        'HTTPS': ['Cloud Storage', 'Cloud Service', 'User Account']
    }
}

database_services = {
    'DSI': {
        'HTTPS': ['Cloud Storage', 'Cloud Service', 'User Account']
    }
}

def get_machine_list(num_client) -> List[Machine]:
    """Return list of machines to build the environment."""

    client_machines = [
        Client(instance_name='PC_{}'.format(i+2), platforms=['Windows'], connected_machines=['Switch_1'], value=0, data_sources=client_services)
        for i in range(num_client - 2)
        ] + [
            Client(instance_name='PC_1', platforms=['Windows'], connected_machines=['Switch_1'], value=0, is_infected=True, data_sources=client_services,
            outcomes=[
                LeakedMachineIP(machine_ip=['PC_{}'.format(i+1) for i in range(num_client)])
            ])
        ] + [
            Client(instance_name='PC_{}'.format(num_client), platforms=['Windows'], connected_machines=['Switch_1'], value=0, data_sources=client_services,
            outcomes=[
                LeakedCredentials(credentials=[Credential(port='HTTPS', machine='MailServer', profile='DSI')])
                ])
        ]

    plug_machines = [
        Plug(instance_name='Switch_1', platforms=[], connected_machines=['PC_{}'.format(i+1) for i in range(num_client)]+['Router']),
        Plug(instance_name='Switch_2', platforms=[], connected_machines=['DatabaseServer', 'MailServer', 'CommunicationServer', 'Firewall_1']),
        Plug(instance_name='Router', platforms=[], connected_machines=['Switch_1', 'Firewall_1', 'Firewall_2'])
        ]
    
    internal_servers = [
        Server(instance_name='DatabaseServer', platforms=['Linux'], connected_machines=['Switch_2'], value=1000, data_sources=database_services,
        outcomes=[Collection(data='Confidential document', required_right=UserRight.LOCAL_USER, absolute_value=1000)]), 
        Server(instance_name='MailServer', platforms=['IaaS'], connected_machines=['Switch_2'], value=200, data_sources=servermail_services,
        outcomes=[LeakedMachineIP(machine_ip=[
            'DatabaseServer', 'CommunicationServer', 'GoogleDrive'], required_right=UserRight.LOCAL_USER)
            ]),
        Server(instance_name='CommunicationServer', platforms=['PRE'], connected_machines=['Switch_2'], value=200, data_sources=servermail_services) 
    ]

    external_servers = [
        Cloud(instance_name='GoogleDrive', platforms=['Google Workspace'], connected_machines=['Firewall_2'], value=500, data_sources=googledrive_services,
        outcomes=[LeakedCredentials(credentials=[Credential(port='HTTPS', machine='DatabaseServer', profile='DSI')])])
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
    
    return machine_list


def get_little_environment_profiles(num_client) -> EnvironmentProfiles:
    """Return the environment profiles."""
    profiles = {
        Dev(): num_client - 2,        
        DSI(): 1
    }
    machine_list = get_machine_list(num_client)

    return EnvironmentProfiles(profiles, machine_list)
    

def get_little_environment_network(num_client) -> Network:
    """Return the network."""
    machine_list = get_machine_list(num_client)

    return Network(machine_list, name='Little_environment')
