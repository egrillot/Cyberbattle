

import numpy as np

from typing import List
from .functions import kahansum


class MarkovProcess:
    """MarkovProcess class allowing us to create activities whose dynamics follow markov process models."""

    def __init__(self, states: List[object], initial_distribution: np.ndarray, transition_matrix: np.ndarray) -> None:
        """Init the MarkovProcess.
        
        Input:
        states: it refers to the state names (List[object])
        initial_distribution: defines the distribution law to choose the first state starting this markov process (np.ndarray)
        transition_matrix: each row will define the distribution law to go to the next state from the current state (np.ndarray).
        """
        states_count = len(states)
        initial_distribution_shape = initial_distribution.shape
        transition_matrix_shapes = transition_matrix.shape

        if len(initial_distribution_shape) != 1:
            raise ValueError('The provided initial distribution has shape of length {} instead of shape of length 1.'.format(len(initial_distribution_shape)))
        
        if initial_distribution_shape[0] != states_count:
            raise ValueError('The provided initial distribution is a {} dimensional vector but it requires a {} dimensional vector'.format(initial_distribution_shape[0], states_count))

        if kahansum(initial_distribution) != 1:
            raise ValueError("The sum over the initial distribution : {} isn't equal to 1.".format(initial_distribution))
        
        if len(transition_matrix_shapes) != 2:
            raise ValueError('The provided transition matrix has shape of length {} instead of shape of length 2.'.format(len(transition_matrix_shapes)))
        
        if transition_matrix_shapes != (states_count, states_count):
            raise ValueError('The provided transition matrix is a {} dimensional vector but it requires a {} dimensional vector.'.format(transition_matrix_shapes, (states_count, states_count)))

        for i in range(states_count):

            if kahansum(transition_matrix[i, :]) != 1:
                raise ValueError("The sum over the row {} in the transition matrix isn't equal to 1".format(i))
        
        self.states = np.array(states)
        self.components = np.arange(states_count)
        self.initial_distribution = initial_distribution
        self.transition_matrix = transition_matrix
        self.last_call = None
    
    def call(self) -> object:
        """Return a state with respect to the markov process."""
        index = np.random.choice(self.components, p=self.transition_matrix[self.last_call, :]) if self.last_call else np.random.choice(self.components, p=self.initial_distribution)
        self.last_call = index

        return self.states[index]
    
    def get_states(self) -> List[object]:
        """Return the state list."""
        return self.states
    
    def reset(self) -> None:
        """Reset the Markov process."""
        self.last_call = None


class MultiMarkovProcess:
    """MultiMarkovProcess class."""

    def __init__(self, markov_process_list: List[MarkovProcess], markov_process_transition: np.ndarray) -> None:
        """Init the MultiMarkovProcess.
        
        Input:
        markov_process_list: list of the different markov process (List[MarkovProcess])
        markov_process_transition: array where each row corresponds to the distribution law of moving from one Markov chain to another (np.ndarray).
        """
        markov_process_transition_shape = markov_process_transition.shape

        if len(markov_process_transition_shape) != 2:
            raise ValueError(f"The provided markov process transition matrix is {markov_process_transition_shape}-dimensional but must be 2-dimensional.")

        n_chain = len(markov_process_list)

        if markov_process_transition_shape != (n_chain, n_chain):
            raise ValueError(f"The provided markov process transition matrix shape is {markov_process_transition_shape} but {n_chain} Markov process have been provide.")

        for i in range(n_chain):

            if kahansum(markov_process_transition[i, :]) != 1:
                raise ValueError("The sum over the row {} in the transition matrix isn't equal to 1".format(i))

        self.markov_process_list = markov_process_list
        self.markov_process_transition = markov_process_transition
        self.components = np.arange(n_chain)
        self.last_call = None
    
    def get_markov_process_list(self) -> List[MarkovProcess]:
        """Return the markov process list."""
        return self.markov_process_list
    
    def call(self) -> object:
        """Return a state with respect to the markov process."""
        if not self.last_call:
            
            chain_index = np.random.choice(self.components)
        
        else:

            chain_index = np.random.choice(self.components, p=self.markov_process_transition[self.last_call, :])
            
        self.last_call = chain_index

        return self.markov_process_list[chain_index].call()
    
    def reset(self) -> None:
        """Reset the multi markov process."""
        self.last_call = None

        for markov_process in self.markov_process_list:

            markov_process.reset()
