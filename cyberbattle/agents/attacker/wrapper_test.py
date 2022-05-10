


from .wrapper import Wrapper, InfectedMachines
from ...env.samples.little_network import get_little_environment_network, DSI, Dev
from ..battle_environment import CyberBattleEnv

import numpy as np

num_client = 5

network = get_little_environment_network(num_client)
profiles = {
    DSI(num_client): 1,
    Dev(num_client): num_client - 1
}

env = CyberBattleEnv(network, profiles)
bounds = env.get_attacker_bounds()

features_list_1 = ['All']
features_list_2 = [InfectedMachines(bounds)]


def features_init_test():

    wrapper_1 = Wrapper(
        features_list_1,
        bounds
    )

    wrapper_2 = Wrapper(
        features_list_2,
        bounds
    )

def interaction_test():

    wrapper = Wrapper(
        features_list_1,
        bounds
    )

    observation_1 = wrapper.observation(env) 
    expected_observation_1: np.ndarray = np.array([[0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    assert observation_1.shape == expected_observation_1.shape
    assert np.count_nonzero(observation_1 - expected_observation_1) == 0

    attack = {'local': np.array([0, 0])}
    reward, _ = env.attacker_step(attack)
    wrapper.process_result(reward, attack)

    observation_2 = wrapper.observation(env) 
    expected_observation_2: np.ndarray = np.array([[0, 0, 1, 5, 4, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
       [1, 0, 1, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [2, 0, 1, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [3, 0, 1, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
       [4, 0, 1, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])

    assert observation_2.shape == expected_observation_2.shape
    assert np.count_nonzero(observation_2 - expected_observation_2) == 0

    attack = {'local': np.array([0, 2])}
    reward, _ = env.attacker_step(attack)
    wrapper.process_result(reward, attack)

    observation_3 = wrapper.observation(env) 
    expected_observation_3: np.ndarray = np.array([[0, 0, 1, 5, 4, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
        0, 0, 0, 0],
       [1, 0, 1, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0],
       [2, 0, 1, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0],
       [3, 0, 1, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0],
       [4, 0, 1, 5, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0]])

    assert observation_3.shape == expected_observation_3.shape
    assert np.count_nonzero(observation_3 - expected_observation_3) == 0

    attack = {'remote': np.array([0, 4, 3])}
    reward, _ = env.attacker_step(attack)
    wrapper.process_result(reward, attack)

    observation_4 = wrapper.observation(env) 
    expected_observation_4: np.ndarray = np.array([[0, 1, 1, 6, 5, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
        0, 0, 0, 0],
       [1, 1, 1, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0],
       [2, 1, 1, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0],
       [3, 1, 1, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0],
       [4, 1, 1, 6, 5, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0],
       [5, 1, 1, 6, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0]])

    assert observation_4.shape == expected_observation_4.shape
    assert np.count_nonzero(observation_4 - expected_observation_4) == 0

    attack = {'connect': np.array([0, 0])}
    reward, _ = env.attacker_step(attack)
    wrapper.process_result(reward, attack)

    observation_5 = wrapper.observation(env) 
    expected_observation_5: np.ndarray = np.array([[0, 1, 2, 6, 4, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1,
        0, 0, 1, 0],
       [1, 1, 2, 6, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0],
       [2, 1, 2, 6, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0],
       [3, 1, 2, 6, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0],
       [4, 1, 2, 6, 4, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        1, 0, 0, 0],
       [5, 1, 2, 6, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0]])

    assert observation_5.shape == expected_observation_5.shape
    assert np.count_nonzero(observation_5 - expected_observation_5) == 0

    env.reset()
    wrapper.reset()
    observation_6 = wrapper.observation(env)
    
    assert np.count_nonzero(observation_6 - expected_observation_1) == 0