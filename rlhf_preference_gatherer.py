import abc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import torch as th
import tkinter as tk

import discrete_graphics as gr


class PreferenceGatherer(abc.ABC):
    """Base class for gathering preference comparisons between trajectory pairs."""

    def __init__(
        self,
        animations
    ) -> None:
        """Initializes the preference gatherer."""

    @abc.abstractmethod
    def __call__(self, trajectory_pairs) -> np.ndarray:
        """Gathers the preference for each pair of trajectory in 'trajectory_pairs'.

        Args:
            trajectory_pairs: sequence of pairs of trajectories

        Returns:
            A numpy array with shape (b, ), where b is the length of the input
            (i.e. batch size). Each item in the array is the preference score
            for the corresponding pair of trajectories.

            Preference score: 
                1 better? => 1
                2 better? => 0
                Equality? => 0.5
                Don't know? => not taken into account (only relevant with humans)
        """ 


class SyntheticPreferenceGatherer(PreferenceGatherer):
    def __init__(self, coeff, gamma, device):
        self.coeff = coeff
        self.gamma = gamma
        self.device = device

    def __call__(self, trajectory_pairs):
        """
        Gathers the synthetic preference for each pair of trajectory in 'trajectory_pairs'.

        Args:
            trajectory_pairs: A list of tuples, where each tuple contains a list of two trajectories.

        Returns:
            A numpy array with shape (b,), where b is the length of the input (i.e., batch size).
            Each item in the array is the preference score for the corresponding pair of trajectories.
            Preference score:
                1 better? => 1
                2 better? => 0
                Equality? => 0.5
        """
        preferences = th.tensor([], device=self.device, dtype=th.float32)

        for trajectory_pair in trajectory_pairs:
            trajectory1 = trajectory_pair[0].get_transitions()
            trajectory2 = trajectory_pair[1].get_transitions()

            # Calculate reward for both trajectories
            reward1 = self.reward_trajectory(trajectory1)
            reward2 = self.reward_trajectory(trajectory2)

            # Compute preference scores
            if reward1 > reward2:
                preferences = th.cat([preferences, th.tensor([1.], device=self.device, dtype=th.float32)])
            elif reward1 < reward2:
                preferences = th.cat([preferences, th.tensor([0.], device=self.device, dtype=th.float32)])
            else:
                preferences = th.cat([preferences, th.tensor([0.5], device=self.device, dtype=th.float32)])

        return preferences

    def reward_trajectory(self, trajectory):
        """Compute reward for a trajectory."""
        reward = 0
        for i, transition in enumerate(trajectory):
            reward += self.gamma**i * self.reward_transition(transition)
        return reward
    
    def reward_transition(self, transition):
        return th.dot(self.coeff, transition.reward_features)
    

class HumanPreferenceGatherer(PreferenceGatherer):
    def __init__(self):
        "Initialises Preference Gatherer"

    def __call__(self, trajectory_pairs):
        """
        Gathers the human  preference for each pair of trajectory in 'trajectory_pairs'.

        Args:
            trajectory_pairs: A list of tuples, where each tuple contains two trajectories.
                Each trajectory is a list of transitions.

        Returns:
            A numpy array with shape (b,), where b is the length of the input (i.e., batch size).
            Each item in the array is the preference score for the corresponding pair of trajectories.
            Preference score:
                1 better? => 1
                2 better? => 0
                Equality? => 0.5
        """
        preferences = th.tensor([])

        for trajectory_pair in trajectory_pairs:
            # Retrieve animations
            anim1, fig1, ax1 = trajectory_pair[0].get_animation()
            # print(f"Number {fig1.number}")
            ani1 = gr.animate(fig1, anim1)
            fig1.show()

            # plt.show()

            anim2, fig2, ax2 = trajectory_pair[1].get_animation()
            print(f"Number {fig2.number}")
            print(f"Number {fig1.number}")
            ani2 = gr.animate(fig2, anim2)
            print(f"ANI1: {ani1}")
            print(f"ANI2: {ani2}")

            fig2.show()

            window = tk.Tk()
            greeting = tk.Label(text='hello')
            greeting.pack()
            window.mainloop()

            # [ax1.add_artist(artist) for artist in anim1[-1]]

            # print(f"FIG: {fig1}")
            # print(f"FIG: {fig2}")
            # print(f"FIG: {fig1==fig2}")

            # anim1 = trajectory_pair[0].get_animation()
            # interactive(True)
            # plt.show()
            # print(f"PLT: {plt.get_fignums()}")
            # fig, ax = plt.subplots(1,1)

            # anim2 = trajectory_pair[0].get_animation()

            print(f"PLT: {plt.get_fignums()}")

            #plt.show()

            break 

        plt.close('all')
        return np.array(preferences)
    