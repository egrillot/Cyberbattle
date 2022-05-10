

import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict
from ...agent import Agent
from ..wrapper import Feature, Wrapper
from ...battle_environment import AttackerBounds, CyberBattleEnv
from ....utils.functions import random_argmax


class QMatrix:
    """QMatrix class.
    
    This class allows us to work with a q matrix, update its values and make regressions.
    """

    def __init__(self, n_states: int, n_actions: int, loss: str, qmatrix: np.ndarray=None) -> None:
        """Init withe action space and state space dimensions."""
        self.n_states = n_states
        self.n_actions = n_actions
        self.q: np.ndarray = qmatrix if qmatrix is not None else np.zeros((self.n_actions, self.n_states), dtype=float)
        self.q_count_updates = np.zeros((self.n_actions, self.n_states), dtype=float)
        
        if loss == 'squared error':

            self.loss_function = lambda x: x ** 2
        
        elif loss == 'absolute error':

            self.loss_function = lambda x: np.abs(x)
        
        else:
            raise ValueError(f"Unidefined loss function : {loss}.")

        self.loss: List[float] = []
    
    def reset(self) -> None:
        """Reset qmatrix values."""
        self.q = np.zeros((self.n_actions, self.n_states), dtype=float)
        self.q_count_updates = np.zeros((self.n_actions, self.n_states), dtype=float)
    
    def update(self, current_state: int, action: int, next_state: int, reward: float, gamma: float, learning_rate: float) -> None:
        """Update the q matrix value with respect to the Bellman's equation and the provided parameters.
        
        Inputs:
        current_state: the current state index (int)
        action: action that had been done given the current state (int)
        next_state: the state that follows the action (int)
        reward: the obtained reward performing the action (float)
        gamma: the Bellman's equation constant gamma (float)
        learning_rate: the Bellman's equation constant learning_rate (float).
        """
        expected_reward, _ = self.exploit(next_state)

        # Bellman equation
        delta_time = reward + gamma * expected_reward - self.q[action, current_state]
        self.q[action, current_state] += learning_rate * delta_time
        self.q_count_updates[action, current_state] += 1

        self.loss.append(self.loss_function(delta_time))

    def exploit(self, current_state: int) -> Tuple[float, int]:
        """Return the expected reward and suggested action given the current_state with respect to the q matrix values."""
        return random_argmax(self.q[:, current_state])


class QlearnerAction:
    """QlearnerAction class.
    
    This class allow us to save what happened at the last step.
    """

    def __init__(self, state: int, action: int, source: int) -> None:
        """Init."""
        self.state = state
        self.action = action
        self.source = source


class Qlearner(Agent):
    """Qlearner agent class.
    
    This model will estimate 2 q-matrix. The first will estimate rewards for each actions and 
    the second predicts the best source for the attack if a remote attack is sugested by the first one.
    """

    def __init__(
        self,
        bounds: AttackerBounds,
        features: List[str] or List[Feature],
        already_trained_q: np.ndarray=None,
        loss: str= 'squared error',
        gamma: float=0.025,
        learning_rate: float=0.01,
        hash_size: int=10000
    ) -> None:
        """Init.
        
        Inputs:
        """
        name = 'Qlearner'
        type = 'learner'

        self.wrapper = Wrapper(features, bounds, hash_size)
        self.n_states = self.wrapper.states_flat_size()
        self.n_actions = bounds.maximum_local_attack + bounds.maximum_remote_attack * bounds.maximum_machines_count + bounds.maximum_credentials_count + 1
        self.qmatrix = QMatrix(self.n_states, self.n_actions, qmatrix=already_trained_q, loss=loss)

        self.learning_rate = learning_rate
        self.gamma = gamma

        self.last_action = None

        super().__init__(name, type)
    
    def qmatrix_values(self) -> np.ndarray:
        """Return the q matrix values."""
        return self.qmatrix.q

    def save(self, path_directory: str) -> None:
        """Save the matrix values as pickle file."""
        file_name = f'{path_directory}/{self.name}_qmatrix.pkl' if path_directory[-1] != '/' else f'{path_directory}{self.name}_qmatrix.pkl'

        with open(file_name, 'wb') as f:

            pkl.dump(((self.gamma, self.learning_rate, self.wrapper), self.qmatrix))
        f.close()
    
    def get_descritpion(self) -> str:
        """Return the parameters."""
        return f"n_states={self.n_states}, n_actions={self.n_actions}, learning_rate={self.learning_rate}, gamma={self.gamma}."

    def exploit(self, env: CyberBattleEnv) -> Tuple[bool, Dict[str, np.ndarray]]:
        """Exploit both q matrix.
        
        This method returns whether the choosen action is executable on the environment or not and the choosen action.
        """
        states = self.wrapper.states_observed(env)
        
        current_reward = - np.infty

        for source, state in enumerate(states):

            expected_reward, action = self.qmatrix.exploit(state)

            if expected_reward > current_reward:
                
                kept_state = state
                kept_action = action
                kept_source = source
                current_reward = expected_reward
        
        self.last_action = QlearnerAction(state=kept_state, action=kept_action, source=kept_source)

        attacker_action = self.wrapper.model2action(kept_action, kept_source)

        if env.is_attacker_action_valid(attacker_action):

            return True, attacker_action
        
        self.qmatrix.update(current_state=kept_state, action=kept_action, next_state=kept_state, reward=0, gamma=self.gamma, learning_rate=self.learning_rate)
        
        return False, attacker_action
    
    def explore(self, env: CyberBattleEnv) -> Dict[str, np.ndarray]:
        """Explore the environment."""
        attack = env.sample_random_valid_action_attacker()
        action, source = self.wrapper.action2model(attack)
        state = self.wrapper.states_observed_at(env, source)
        self.last_action = QlearnerAction(state, action, source)

        return attack

    def learn(self, env: CyberBattleEnv, reward: float) -> None:
        """Update the qmatrix value with respect to the action done before and what happened in the environment."""
        state = self.wrapper.states_observed_at(env, self.last_action.source)
        self.qmatrix.update(
            current_state=self.last_action.state,
            action=self.last_action.action,
            next_state=state,
            reward=reward,
            gamma=self.gamma,
            learning_rate=self.learning_rate
        )

    def loss(self) -> np.ndarray:
        """Return the loss history."""
        return np.array(self.qmatrix.loss)
    
    def new_episode(self) -> None:
        """Reset the loss tracking."""
        self.qmatrix.loss = []
    
    def plot_q_updates_history(self) -> None:
        """Display the updating count for each couple of state-action."""
        qmatrix_updates = self.qmatrix.q_count_updates
        factor_scale = self.n_states // self.n_actions
        qmatrix_updates_rescale = np.zeros((self.n_actions * factor_scale, self.n_states))

        for i in range(self.n_actions):

            for j in range(i, i + factor_scale):

                qmatrix_updates_rescale[j, :] = qmatrix_updates[i, :]

        plt.figure(figsize=(10, 30))
        plt.title('QMatrix updates count')
        plt.imshow(qmatrix_updates_rescale, cmap='hot')
        plt.show()
