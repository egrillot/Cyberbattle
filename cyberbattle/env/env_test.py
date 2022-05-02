"""This file is used to test the different networks in the baseline and the functions/objects in the env directory."""

from typing import List
from .samples.little_network import get_little_environment_network, get_little_environment_profiles
from .utils.data import Data_source
from .utils.machine import Machine

num_client = 4
network = get_little_environment_network(num_client)


def test_little_environment_init():

    #net.display(annotations=True)
    paths: List[Machine] = network.get_paths()

    instance_name_to_index = network.instance_name_to_index
    instance_name_to_machine = network.instance_name_to_machine

    idx1 = instance_name_to_index['PC_1']
    idx2 = instance_name_to_index['DatabaseServer']
    idx3 = instance_name_to_index['GoogleDrive']

    switch_1 = instance_name_to_machine['Switch_1']
    switch_2 = instance_name_to_machine['Switch_2']

    firewall_1 = instance_name_to_machine['Firewall_1']
    firewall_2 = instance_name_to_machine['Firewall_2']

    p1 = [m.get_instance_name() for m in paths[idx1, idx2]]
    p2 = [m.get_instance_name() for m in paths[idx1, idx3]]

    assert p1 == ['PC_1', 'Switch_1', 'Router', 'Firewall_1', 'Switch_2', 'DatabaseServer']
    assert p2 == ['PC_1', 'Switch_1', 'Router', 'Firewall_2', 'GoogleDrive']
    assert len(switch_1.get_connected_machines()) > 1
    assert len(switch_2.get_connected_machines()) > 1
    assert len(firewall_1.get_connected_machines()) == 2
    assert len(firewall_2.get_connected_machines()) == 2
    assert set(network.get_services()) == set(['HTTPS', 'sudo'])
    assert set(network.get_available_datasources()) == set(['User Account',
                                                        'Logon Session',
                                                        'File',
                                                        'Script',
                                                        'Cloud Service',
                                                        'Process',
                                                        'Driver',
                                                        'Cloud Storage'])

def test_data_sources():

    data_sources = Data_source.__subclasses__()

    for ds in data_sources:

        a = ds()
        ds_name = a.get_data_source()
        actions = [a.call() for _ in range(10)]
        assert len(actions) == 10
        assert sum([1 for action in actions if ds_name in action]) == 10

def test_little_environment_profiles_init():

    env_profiles = get_little_environment_profiles(num_client)

    assert env_profiles.nb_profile == num_client
    activities =  env_profiles.on_step(network)

    assert len(activities) == num_client

