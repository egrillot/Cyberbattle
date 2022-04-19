"""provide an environment with 3 internal servers and 1 external server (github projects) with as many as desired."""


from typing import List
from ..utils import flow, user, machine, network
from ...vulnerabilities import outcomes

def get_machine_list(num_client) -> List[machine.Machine]:
    """Return list of machines to build the environment."""

    client_machines = [
        machine.Client(instance_name='PC_{}'.format(i+2), platforms=['Windows'], connected_machines=['Switch_1'], value=0)
        for i in range(num_client - 2)
        ] + [
            machine.Client(instance_name='PC_1', platforms=['Windows'], connected_machines=['Switch_1'], value=0, is_infected=True, outcomes=[
                outcomes.Escalation(userright=flow.UserRight.ADMIN),
                outcomes.LeakedMachineIP(machine_ip=['CommunicationServer'])
            ])
        ] + [
            machine.Client(instance_name='PC_{}'.format(num_client), platforms=['Windows'], connected_machines=['Switch_1'], value=0, outcomes=[
                outcomes.LateralMove()
            ])
        ]

    plug_machines = [
        machine.Plug(instance_name='Switch_1', platforms=[], connected_machines=['PC_{}'.format(i+1) for i in range(num_client)]+['Router']),
        machine.Plug(instance_name='Switch_2', platforms=[], connected_machines=['DatabaseServer', 'MailServer', 'CommunicationServer', 'Firewall_1']),
        machine.Plug(instance_name='Router', platforms=[], connected_machines=['Switch_1', 'Firewall_1', 'Firewall_2'])
        ]
    
    internal_servers = [
        machine.Server(instance_name='DatabaseServer', platforms=['IaaS'], connected_machines=['Switch_2'], value=1000, outcomes=[
            outcomes.Reconnaissance('confidential folder', 100),
            outcomes.LeakedCredentials(credentials=[
                flow.Credential(port='HTTPS', machine='GoogleDrive', cred='DSI_password')
            ])
        ]),
        machine.Server(instance_name='MailServer', platforms=['PRE'], connected_machines=['Switch_2'], value=200),
        machine.Server(instance_name='CommunicationServer', platforms=['Pre'], connected_machines=['Switch_2'], value=200)
    ]

    external_servers = [
        machine.Cloud(instance_name='GoogleDrive', platforms=['Google Workspace'], connected_machines=['Firewall_2'], value=500)
    ]

    firewalls = [
        machine.Firewall(
            instance_name='Firewall_1',
            platforms=[],
            connected_machines=['Router', 'Switch_2'],
            incomings=[
                flow.Traffic(port='HTTPS', rule=flow.Rule.ALLOWED),
                flow.Traffic(port='HTTP', rule=flow.Rule.ALLOWED),
                flow.Traffic(port='SSH', rule=flow.Rule.ALLOWED)
            ],
            outgoings=[
                flow.Traffic(port='HTTPS', rule=flow.Rule.ALLOWED),
                flow.Traffic(port='HTTP', rule=flow.Rule.ALLOWED),
                flow.Traffic(port='SSH', rule=flow.Rule.ALLOWED)
            ]
        ),
        machine.Firewall(
            instance_name='Firewall_2',
            platforms=[],
            connected_machines=['Router', 'GoogleDrive'],
            incomings=[
                flow.Traffic(port='HTTPS', rule=flow.Rule.ALLOWED),
                flow.Traffic(port='HTTP', rule=flow.Rule.ALLOWED),
                flow.Traffic(port='SSH', rule=flow.Rule.ALLOWED)
            ],
            outgoings=[
                flow.Traffic(port='HTTPS', rule=flow.Rule.ALLOWED),
                flow.Traffic(port='HTTP', rule=flow.Rule.ALLOWED),
                flow.Traffic(port='SSH', rule=flow.Rule.ALLOWED)
            ]
        )
    ]

    machine_list: List[machine.Machine] = client_machines + plug_machines + internal_servers + external_servers + firewalls

    for i, m in enumerate(machine_list):

        m.set_ip_adress(i)
    
    return machine_list


def get_little_environment_profiles(num_client) -> user.EnvironmentProfiles:
    """Return the environment profiles."""
    profiles = {
        user.DSI(based_on=['PC', 'Cloud', 'Server']): 1,
        user.Dev(based_on=['PC', 'Cloud']): num_client - 2
    }
    machine_list = get_machine_list(num_client)

    return user.EnvironmentProfiles(profiles, machine_list)
    

def get_little_environment_network(num_client) -> network.Network:
    """Return the network."""
    machine_list = get_machine_list(num_client)

    return network.Network(machine_list, name='Little_environment')

