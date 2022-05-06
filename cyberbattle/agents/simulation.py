


import time
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from typing import Tuple, List, Dict
from .agent import Agent
from .battle_environment import CyberBattleEnv
from .defender.baseline.scan import Scaner
from ..utils.functions import exponential_espilon_decrease
from ..utils.io_utils import ProgressBar, plot_epsilon_greedy_search_result_attacker_alone


training_methods = ['epsilon greedy search']
simulation_types = ['Agent vs Nothing', 'Agent vs Scaner', 'Adversial', 'Custom against Agent']  

class Simulation:
    """Simulation class.
    
    This class allow us to define how we want to train our agents and whether we want to train both agents at the same time or not.
    """

    def __init__(
        self,
        environment: CyberBattleEnv
    ) -> None:
        """Init.
        
        max_iteration: number of maximal iteration by epoch (int)
        attacker: define the agent that will be optimize for the kill chain (Agent)
        """
        self.environment = environment
        self.reset()
    
    def reset(self) -> None:
        """Reset the simulation envrionment."""
        self.environment.reset()
        self.simulations_result: Dict[int, Tuple[List[List[float]], List[Dict]]] = dict()
        self.simulations_description: Dict[int, str] = dict()
        self.nb_simulation = 1
        self.simulation_type = None
        self.training_method = None
        self.trained_agents = dict()

    def epsilon_greedy_search_solo_attacker(
        self,
        attacker: Agent,
        max_iteration: int,
        epochs: int,
        epsilon: float,
        decrease_function,
        defender: Scaner=None,
    ) -> None:
        """Execute an epsilon greedy search to only train one agent as the attacker"""
        if epsilon < 0 or epsilon > 1:
            raise ValueError(f"The provided epsilon value is equal to {epsilon} instead of being between [0, 1].")

        if defender:
            print(f"Training the agent {attacker.get_name()}\nwith parameters : {attacker.get_descritpion()}\nagainst the agent {defender.get_name()}\nwith parameters : {defender.get_descritpion()}.")
            self.simulations_description[self.nb_simulation] = f"Involved agents : {attacker.get_name()} VS {defender.get_name()}, environment : {self.environment.get_name()}, training_method : {self.training_method} with parameters : max_iteration={max_iteration}, epochs={epochs} and epsilon={epsilon}."
        
        else:
            print(f"Training the agent {attacker.get_name()}\nwith parameters : {attacker.get_descritpion()}\nagainst nothing.")
            self.simulations_description[self.nb_simulation] = f"Involved agent : {attacker.get_name()}, environment : {self.environment.get_name()}, training_method : {self.training_method} with parameters : max_iteration={max_iteration}, epochs={epochs} and epsilon={epsilon}."
        
        print(f"Epsilon greedy search parameters : max_iteration={max_iteration}, epochs={epochs} and epsilon={epsilon}.\n")

        all_epochs_rewards = []
        all_history = []
        step_count = 1

        for epoch in range(1, epochs + 1):

            print(f"Epoch : {epoch}/{epochs}, epsilon = {epsilon}")

            self.environment.reset()
            total_reward = 0.0
            all_rewards = []
            
            history = {
                'exploit': {
                    'local': {
                        'successfull': 0,
                        'failed': 0
                        },
                    'remote': {
                        'successfull': 0,
                        'failed': 0
                        },
                    'connect': {
                        'successfull': 0,
                        'failed': 0
                            }                
                },
                'explore': {
                    'local': {
                        'successfull': 0,
                        'failed': 0
                        },
                    'remote': {
                        'successfull': 0,
                        'failed': 0
                        },
                    'connect': {
                        'successfull': 0,
                        'failed': 0
                        },               
                },
                'submarine': 0,
                'exploit -> explore': 0
            }

            successfull_action = 0
            failed_action = 0

            bar = ProgressBar(target = max_iteration)

            for iteration in range(1, max_iteration + 1):

                p = np.random.random()

                if p <= epsilon:

                    action = attacker.explore(self.environment)
                    action_type = 'explore'
                
                else:

                    performable, action = attacker.exploit(self.environment)
                    action_type = 'exploit'
                    
                    if not performable:

                        action = attacker.explore(self.environment)
                        action_type = 'exploit -> explore'

                env_done, reward = self.environment.attacker_step(action)
                decision = list(action.keys())[0]

                if decision == 'submarine':

                    history[decision] += 1
                
                else:

                    if reward >=0:

                        result = 'successfull'
                        successfull_action += 1
                    
                    else:

                        result = 'failed'
                        failed_action += 1

                    if action_type != 'exploit -> explore':

                        history[action_type][decision][result] += 1
                    
                    else:

                        history[action_type] += 1
                
                attacker.learn()
                total_reward += reward
                values = {
                    'cumulate rewards': total_reward,
                    'sucessfull action': successfull_action,
                    'failed action': failed_action,
                    'infected machine count': len(self.environment.get_infected_machine())
                }
                bar.update(values, iteration-1)
                all_rewards.append(reward)

                if env_done:

                    break
                
                if defender is not None:
                    
                    _, action = defender.exploit(self.environment)
                    self.environment.defender_step(action)
                
                epsilon = decrease_function(epsilon, step_count)
                step_count += 1
        
            all_epochs_rewards.append(all_rewards)
            all_history.append(history)
            bar.update(values, max_iteration)
            print(f"\nTotal reward : {total_reward}, epoch ended at {iteration} iterations.\n\n###################\n")

        self.simulations_result[self.nb_simulation] = (all_epochs_rewards, all_history)
        self.trained_agents[self.nb_simulation] = (attacker, None)
        self.nb_simulation += 1
        
    def compile(
        self,
        training_method: str='epsilon greedy search',
        simulation_type: str='Agent vs Nothing'
    ) -> None:
        """Compile to init parameters and expectations for the simulation run.
        
        Inputs:
        training_method: specifies what policy the agents must follow to learn how to optimize their decisions, default value: 'epsilon greedy search' (str)
        simulation_type: indicates if two agents are interacting in the environment to defend and attack or not or if the attacker or defender are struggling alone against an iterative algorithm or nothing, default value : 'Agent vs Nothing' which means that the user must indicate which agent he wants to be trained in an environment without defender (str).
        """        
        if not training_method in training_methods:
            raise ValueError(f"The provided training method {training_method} isn't defined, please use a method among those: {training_methods}")
        
        if not simulation_type in simulation_types:
            raise ValueError(f"The provided simulation type {simulation_type} isn't defined, please use a type in the following list : {simulation_types}.")

        self.simulation_type = simulation_type
        self.training_method = training_method
    
    def run(
        self,
        max_iteration: int=1000,
        epsilon: float=0.9,
        epochs: int=10,
        decrease_function=exponential_espilon_decrease(epsilon_min=0.01, exponential_decay=1000),
        attacker: Agent=None,
        defender: Agent=None,
        plot_results: bool=True
    ) -> None:
        """Run the simulation to train one or two agents with respect to the compilation.
        
        Input:
        max_iteration: the maximum iteration per simulation, default value : 1000 (int)
        epsilon: if the selected training method is espilon greedy search, it corresponds to the start epsilon, default value : 0.9 (float)
        epochs: how many times the simulation must run, default value : 10 (int)
        decreade_function: if the selected training method is espilon greedy search, it refers to the used function to update the epsilon through iterations, default value is the function exponential_espilon_decrease available in cyberbattle.utils.functions with default parameters epsilon_min=0.01 and exponential_decay=1000 (function)
        attacker: the agent to train as attacker, default value : None (Agent)
        defender: the agent to train as defender, default value : None (Agent)
        plot_results: whether the training plot must be displayed or not, default value : True (bool).
        """
        if not self.simulation_type:
            raise ValueError("Before running the simulation please use method : compile to precise what simulation type : {self.simulation_types} and what training method : {self.training_methods} you want to use.")
        
        if self.simulation_type == 'Agent vs Nothing':

            if self.training_method == 'epsilon greedy search':

                if attacker is None:
                    raise ValueError("Please provide an Agent instance as the attacker parameter to run the simulation.")
                
                if (defender is not None) and (defender.get_type() != 'Scaner'):

                    defender = None
                    print(f"You compiled with a simulation type : {self.simulation_type} so the defender change has been changed to None. Moreover, if you want to put a defence algorithm against the attacker, you can choose to assign to the defender variable an instance of Scaner available in cyberbattle.agents.defender.baseline.scan .")

                self.epsilon_greedy_search_solo_attacker(attacker=attacker, defender=defender, max_iteration=max_iteration, epochs=epochs, epsilon=epsilon, decrease_function=decrease_function)

                if plot_results:

                    plot_epsilon_greedy_search_result_attacker_alone(
                        simulation_result=self.simulations_result[self.nb_simulation - 1],
                        simulation_description=self.simulations_description[self.nb_simulation - 1]
                    )

                    print("Last attack details :\n")
                    self.environment.display_attacker_history()
