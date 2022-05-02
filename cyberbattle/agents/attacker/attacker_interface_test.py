

from ..battle_environment import CyberBattleEnv
from ...env.samples.little_network import get_little_environment_network, get_machine_list, Dev, DSI
from ...agents.attacker.attacker_interface import AttackerGoal, Attacker

num_client = 5
net = get_little_environment_network(num_client)
machines_list = get_machine_list(num_client)
profiles = {
    DSI(num_client): 1,
    Dev(num_client): num_client - 1
}
env = CyberBattleEnv(net, profiles)


def test_attacker_interface():

    attacker = Attacker(
        goals=AttackerGoal(
            reward=0,
            nb_flag=1
        ),
        attacks=env.id_to_attack,
        network=net,
        attacks_by_machine=env.attacks_by_machine,
        start_time=env.get_start_time()
    )

    assert attacker.is_discovered('PC_1')