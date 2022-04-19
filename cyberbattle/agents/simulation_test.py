"""Test if the simulation works on each environment samples."""

from .battle_environment import CyberBattleEnv
from ..vulnerabilities.attacks import AttackSet
from ..env.samples import little_network

num_client = 4


def test_simulation_little_network_init():

    net = little_network.get_little_environment_network(num_client)
    env_profiles = little_network.get_little_environment_profiles(num_client)
    machines_list = little_network.get_machine_list(num_client)
    attacks = AttackSet(machines_list, env_profiles)
    battle_env = CyberBattleEnv(net, env_profiles, attacks)
