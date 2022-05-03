

import numpy as np

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

    attack = {'local': np.array([0, 0])}
    reward, _ = attacker.on_step(attack)

    assert reward == 20
    assert set(attacker.get_discovered_machines()) == set(['PC_2', 'PC_1', 'PC_3', 'PC_4', 'PC_5'])

    attack = {'local': np.array([0, 3])}
    reward, _ = attacker.on_step(attack) 

    assert reward == -10

    attack = {'local': np.array([0, 3])}
    reward, _ = attacker.on_step(attack)

    assert reward == -12

    attack = {'remote': np.array([0, 4, 1])}
    reward, _ = attacker.on_step(attack)

    assert reward == 25
    assert len(attacker.get_discovered_credentials()) == 1

    cred = attacker.get_discovered_credentials()[0]
    assert cred[0] == 'HTTPS'
    assert cred[1] == 'MailServer'
    assert cred[2] == 'DSI'
    assert set(attacker.get_discovered_machines()) == set(['PC_2', 'PC_1', 'PC_3', 'PC_4', 'PC_5', 'MailServer'])

    attack = {'connect': np.array([0, 0])}
    reward, _ = attacker.on_step(attack)

    assert reward == 250
    assert len(attacker.get_infected_machines()) == 2

    attack = {'local': np.array([5, 3])}
    reward, _ = attacker.on_step(attack)

    assert reward == 13
    assert len(attacker.get_discovered_machines()) == 9

    attack = {'remote': np.array([0, 5, 4])}
    reward, _ = attacker.on_step(attack)

    assert reward == -10

    attack = {'remote': np.array([0, 8, 4])}
    reward, _ = attacker.on_step(attack)

    assert reward == 18
    assert len(attacker.get_discovered_credentials()) == 2

    attack = {'connect': np.array([0, 1])}
    reward, _ = attacker.on_step(attack)

    assert reward == 1050
    assert len(attacker.get_infected_machines()) == 3

    attack = {'submarine'}
    reward, _ = attacker.on_step(attack)

    assert reward == 0

    attack = {'local': np.array([6, 2])}
    reward, _ = attacker.on_step(attack)

    assert reward == 1000

    for cr, expected_cr in zip(attacker.get_cumulative_rewards(), [20, 10, -2, 23, 273, 286, 276, 294, 1344, 2344]):

        assert cr == expected_cr
        
    assert attacker.reached_goals()
    