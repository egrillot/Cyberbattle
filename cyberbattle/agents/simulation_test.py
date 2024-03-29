"""Test if the simulation works on each environment samples."""

from .battle_environment import CyberBattleEnv
from ..env.samples.little_network import get_little_environment_network, get_machine_list, Dev, DSI

num_client = 5
net = get_little_environment_network(num_client)
machines_list = get_machine_list(num_client)
profiles = {
    DSI(num_client): 1,
    Dev(num_client): num_client - 1
}
env = CyberBattleEnv(net, profiles)

#def test_simulation_little_network_init():

