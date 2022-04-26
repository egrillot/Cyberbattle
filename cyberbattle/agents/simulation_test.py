"""Test if the simulation works on each environment samples."""

from .battle_environment import CyberBattleEnv
from ..env.samples.little_network import get_little_environment_network, get_machine_list
from ..env.utils.user import DSI, Dev

num_client = 5
net = get_little_environment_network(num_client)
machines_list = get_machine_list(num_client)
profiles = {
    DSI(): 1,
    Dev(): num_client - 2
}

def test_simulation_little_network_init():

    env = CyberBattleEnv(net, profiles)
    profile_count = env.get_profile_count()
    env.reset()
    for _ in range(10):
        matrix = env.step()
        assert matrix.shape == (profile_count + 1, 5)