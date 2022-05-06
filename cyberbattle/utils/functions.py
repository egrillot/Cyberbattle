

import numpy as np
import math

from typing import Tuple


def kahansum(probability_distribution: np.ndarray, axis: int=0) -> float:
    """Return the kahan sum of the given probability distribution."""
    s = np.zeros(probability_distribution.shape[:axis] + probability_distribution.shape[axis+1:])
    c = np.zeros(s.shape)
    for i in range(probability_distribution.shape[axis]):
        # http://stackoverflow.com/a/42817610/353337
        y = probability_distribution[(slice(None),) * axis + (i,)] - c
        t = s + y
        c = (t - s) - y
        s = t.copy()

    return s


def exponential_espilon_decrease(epsilon_min: float, exponential_decay: float):
    """Return a function to decrease the epsilon during a training with the epsilon greedy search mehtod."""
    def f(epsilon: float, step_count: int) -> float:
        """Return the updated epsilon."""
        return (epsilon - epsilon_min) * math.exp((-1) * step_count / exponential_decay) + epsilon_min
    
    return f