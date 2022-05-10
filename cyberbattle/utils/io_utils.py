

import time
import numpy as np
import matplotlib.pyplot as plt

from typing import Dict, List, Tuple


class ProgressBar:
    """Display a progress bar."""

    def __init__(
        self,
        target: int=None,
        width: int=30,
        interval: float=0.05
    ) -> None:
        """Init the progress bar.
        
        Inputs:
        target: total number of steps expected, None if unknown (int)
        width: progress bar width on screen, defaut value: 30 (int)
        interval: Minimum visual progress update interval in seconds, default value: 0.05 (float).
        """
        self.target = target
        self.width = width
        self.interval = interval
        self._start = time.time()
        self._last_update = 0
        self._time_at_epoch_start = self._start
        self._time_at_epoch_end = None
        self._time_after_first_step = None
        self._total_width = 0

    def update(self, values: Dict[str, float], step: int, finalize: bool=None) -> None:
        """Update the progress bar.
        
        Inputs:
        values: tracked current variables values (Dict[str, float])
        step: current step index (int)
        finalize: whether this is the last update for the progress bar or not. If finalize=None, default value is to current >= self.target (bool).
        """
        if finalize is None:

            if self.target is None:

                finalize = False

            else:

                finalize = step >= self.target

        now = time.time()
        message = ''
        info = ' - %.0fs' % (now - self._start)

        if step == self.target:

            self._time_at_epoch_end = now
        
        if now - self._last_update < self.interval and not finalize:

            return
        
        prev_total_width = self._total_width
        message += '\b' * prev_total_width
        message += '\r'

        if self.target is not None:

            numdigits = int(np.log10(self.target)) + 1
            bar = ('%' + str(numdigits) + 'd/%d [') % (step, self.target)
            prog = float(step) / self.target
            prog_width = int(self.width * prog)

            if prog_width > 0:

                bar += ('=' * (prog_width - 1))

            if step < self.target:

                bar += '>'

            else:

                bar += '='

            bar += ('.' * (self.width - prog_width))
            bar += ']'

        else:

            bar = '%7d/Unknown' % step

        self._total_width = len(bar)
        message += bar
        time_per_unit = self._estimate_step_duration(step, now)

        if self.target is None or finalize:

            info += self._format_time(time_per_unit, 'iteration')

        else:
    
            eta = time_per_unit * (self.target - step)

            if eta > 3600:

                eta_format = '%d:%02d:%02d' % (eta // 3600, (eta % 3600) // 60, eta % 60)

            elif eta > 60:

                eta_format = '%d:%02d' % (eta // 60, eta % 60)
            else:

                eta_format = '%ds' % eta

            info = ' - estimated time remaining: %s' % eta_format
        
        for name, value in values.items():

            info += ' - %s:' % name + ' %s' % value
        
        info += '     '

        self._total_width += len(info)

        if prev_total_width > self._total_width:

            info += (' ' * (prev_total_width - self._total_width))

        if finalize:

            info += '\n'

        message += info
        self._last_update = now
        
        print(message, end='\r')

    def _format_time(self, time_per_unit: float, unit_name: str) -> str:
        """Format a given duration to display to the user.

        Output:
        a string with the correctly formatted duration and units
        """
        formatted = ''

        if time_per_unit >= 1 or time_per_unit == 0:

            formatted += ' %.0fs/%s' % (time_per_unit, unit_name)

        elif time_per_unit >= 1e-3:

            formatted += ' %.0fms/%s' % (time_per_unit * 1e3, unit_name)

        else:

            formatted += ' %.0fus/%s' % (time_per_unit * 1e6, unit_name)

        return formatted

    def _estimate_step_duration(self, current: int=None, now: float=0.0) -> float:
        """Estimate the duration of a single step.

        Inputs:
        current: index of current step, default value is None (int)
        now: the current time, default value is 0.0 (float).
        """
        if current:
            
            if self._time_after_first_step is not None and current > 1:

                time_per_unit = (now - self._time_after_first_step) / (current - 1)
            else:

                time_per_unit = (now - self._start) / current

            if current == 1:

                self._time_after_first_step = now

            return time_per_unit

        else:

            return 0


def plot_epsilon_greedy_search_result_attacker_alone(
    simulation_result: Tuple[List[List[float]], List[Dict]],
    simulation_description: str
) -> None:
    """Plot results of the provided simulation results."""

    all_epochs_rewards, all_history, mean_loss = simulation_result
    n_epoch = len(all_epochs_rewards)
    lengths = [len(epoch) for epoch in all_epochs_rewards]
    longest_epoch_length = max(lengths)
    all_cumulatives_reward = np.zeros((n_epoch, longest_epoch_length), dtype=float)

    fig, ax = plt.subplots(3, 2, figsize=(30, 15))
    fig.suptitle(simulation_description, fontsize=14)

    for i, episode in enumerate(all_epochs_rewards):
        
        episode_length = len(episode)
        episode = np.array(episode)
        paded_episode = np.pad(episode, pad_width=(0, longest_epoch_length - episode_length))
        all_cumulatives_reward[i, :] = np.cumsum(paded_episode)

    avg = np.average(all_cumulatives_reward, axis=0)
    std = np.std(all_cumulatives_reward, axis=0)

    x = [i for i in range(longest_epoch_length)]
    ax[0, 0].plot(x, avg)
    ax[0, 0].fill_between(x, avg - std, avg + std, alpha=0.5)
    ax[0, 0].set_title('Cumulative rewards vs iterations')
    ax[0, 0].set(xlabel='iteration', ylabel='cumulative reward')

    x = [i for i in range(n_epoch)]
    ax[0, 1].plot(x, lengths)
    ax[0, 1].set_title('Duration vs epochs')
    ax[0, 1].set(xlabel='epoch', ylabel='duration')

    all_history_count = np.zeros((n_epoch, 5), dtype=int)

    for i, history in enumerate(all_history):

        all_history_count[i, 0] = history['exploit']['local']['successfull'] + history['exploit']['remote']['successfull'] + history['exploit']['connect']['successfull']
        all_history_count[i, 1] = history['exploit']['local']['failed'] + history['exploit']['remote']['failed'] + history['exploit']['connect']['failed']
        all_history_count[i, 2] = history['explore']['local']['successfull'] + history['explore']['remote']['successfull'] + history['explore']['connect']['successfull']
        all_history_count[i, 3] = history['explore']['local']['failed'] + history['explore']['remote']['failed'] + history['explore']['connect']['failed']
        all_history_count[i, 4] = history['submarine'] + history['submarine'] + history['submarine']

    ax[1, 0].plot(x, all_history_count[:, 0], label='successfull exploit')
    ax[1, 0].plot(x, all_history_count[:, 1], label='failed exploit')
    ax[1, 0].plot(x, all_history_count[:, 2], label='successfull explore')
    ax[1, 0].plot(x, all_history_count[:, 3], label='failed explore')
    ax[1, 0].plot(x, all_history_count[:, 4], label='submarine')
    ax[1, 0].set_title('Success & failed action count by exploration and exploitation')
    ax[1, 0].set(xlabel='epoch', ylabel='count')
    ax[1, 0].legend(loc='lower right')

    all_history_type_rate = np.zeros((n_epoch, 3), dtype=float)

    for i, history in enumerate(all_history):

        all_history_type_rate[i, 0] = 1 if history['exploit']['local']['failed'] == 0 else history['exploit']['local']['successfull'] / ( history['exploit']['local']['failed'] + history['exploit']['local']['successfull'] )
        all_history_type_rate[i, 1] = 1 if history['exploit']['remote']['failed'] == 0 else history['exploit']['remote']['successfull'] / ( history['exploit']['remote']['failed'] + history['exploit']['remote']['successfull'] )
        all_history_type_rate[i, 2] = 1 if history['exploit']['connect']['failed'] == 0 else history['exploit']['connect']['successfull'] / ( history['exploit']['connect']['failed'] + history['exploit']['connect']['successfull'] )

    ax[1, 1].plot(x, all_history_type_rate[:, 0], label='local success rate')
    ax[1, 1].plot(x, all_history_type_rate[:, 1], label='remote success rate')
    ax[1, 1].plot(x, all_history_type_rate[:, 2], label='connect success rate')
    ax[1, 1].set_title('Success rate by action type')
    ax[1, 1].set(xlabel='epoch', ylabel='rate')
    ax[1, 1].legend(loc='lower right')

    ax[2, 0].plot(x, mean_loss)
    ax[2, 0].set_title('Mean loss vs epoch')
    ax[2, 0].set(xlabel='epoch', ylabel='mean loss')
    ax[2, 0].legend(loc='lower right')

    ax[2, 1].plot(x, [history['exploit -> explore'] for history in all_history])
    ax[2, 1].set_title('Exploit to explore vs epoch')
    ax[2, 1].set(xlabel='epoch', ylabel='deflection count')
    ax[2, 1].legend(loc='lower right')

    plt.show()
