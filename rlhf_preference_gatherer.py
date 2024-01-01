"""
@author: elamrani
"""

import abc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
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

        return trajectory_pairs, preferences

    def reward_trajectory(self, trajectory):
        """Compute reward for a trajectory."""
        reward = 0
        for i, transition in enumerate(trajectory):
            reward += self.gamma**i * self.reward_transition(transition)
        return reward
    
    def reward_transition(self, transition):
        return th.dot(self.coeff, transition.reward_features)
    

class HumanPreferenceGatherer(PreferenceGatherer):
    def __init__(self, device):
        "Initialises Preference Gatherer"
        self.preference = None
        self.device = device

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

        def on_button_press(fig1, fig2, window, preference_value):
            # Store the preference value
            self.preference = preference_value

            # Close figures
            plt.close(fig1)
            plt.close(fig2)
            
            # Close the Tkinter window
            window.destroy()
            window.quit()

        valid_trajectory_pairs = []

        for trajectory_pair in trajectory_pairs:
            # Print the list of figure numbers
            # open_figures = plt.get_fignums()
            # print("Open Figures:", open_figures)

            # Retrieve animations
            anim1, fig1, ax1 = trajectory_pair[0].get_animation()
            ani1 = gr.animate(fig1, anim1)

            anim2, fig2, ax2 = trajectory_pair[1].get_animation()
            ani2 = gr.animate(fig2, anim2)

            # Create the Tkinter window
            window = tk.Tk()
            window.title("Matplotlib Animation Window")

            window.geometry("1500x1200")  # Set your desired width and height

            # Create the buttons
            left_button = tk.Button(window, text="TOP", command=lambda: on_button_press(fig1, fig2, window, 1))
            left_button.pack(side=tk.LEFT, padx=10)

            same_button = tk.Button(window, text="SAME", command=lambda: on_button_press(fig1, fig2, window, 0.5))
            same_button.pack(side=tk.LEFT, padx=10)

            right_button = tk.Button(window, text="BOTTOM", command=lambda: on_button_press(fig1, fig2, window, 0))
            right_button.pack(side=tk.LEFT, padx=10)

            not_sure_button = tk.Button(window, text="NOT SURE", command=lambda: on_button_press(fig1, fig2, window, -2))
            not_sure_button.pack(side=tk.LEFT, padx=10)

            # Embed Matplotlib figures in Tkinter window
            canvas1 = FigureCanvasTkAgg(fig1, master=window)
            canvas1.draw()
            canvas1.get_tk_widget().pack(side=tk.TOP, fill=tk.Y, expand=1)

            canvas2 = FigureCanvasTkAgg(fig2, master=window)
            canvas2.draw()
            canvas2.get_tk_widget().pack(side=tk.BOTTOM, fill=tk.Y, expand=1)

            window.mainloop()
            
            print(f"Pref: {self.preference}")
            if self.preference != -2:  # when not sure (e.g. because display bug)
                preferences = th.cat([preferences, th.tensor([self.preference], device=self.device, dtype=th.float32)])
                valid_trajectory_pairs.append(trajectory_pair)

        plt.close('all')
        return valid_trajectory_pairs, preferences
