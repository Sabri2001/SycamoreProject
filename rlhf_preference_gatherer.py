import abc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import torch as th

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

        # TODO: implement human pref

        for trajectory_pair in trajectory_pairs:
            # Retrieve animations
            # plt.ioff()
            anim1, fig1, ax1 = trajectory_pair[0].get_animation()
            print(f"Number {fig1.number}")
            ani1 = gr.animate(fig1, anim1)

            # plt.show()

            anim2, fig2, ax2 = trajectory_pair[1].get_animation()
            fig2.number=2
            print(f"Number {fig2.number}")
            print(f"Number {fig1.number}")
            ani2 = gr.animate(fig2, anim2)
            print(f"ANI1: {ani1}")
            print(f"ANI2: {ani2}")

            # fig = plt.figure()
            # sfigs = fig.subfigures(2,1)
            # sfigs[0] = fig1
            # sfigs[1] = fig2

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

            # interactive(False)
            plt.show()

            # anim2, fig2, ax2 = trajectory_pair[1].get_animation()
            # animate2 = gr.animate(fig2, anim2)
            # fig2.show()

            # anim2, fig2, ax2 = trajectory_pair[1].get_animation()
            # anim2.show()
            # anim2, fig2, ax = trajectory_pair[1].get_animation()

            # buf = io.BytesIO()
            # pickle.dump(anim1, buf)
            # buf.seek(0)
            # new_anim1 = pickle.load(buf)

            # new_fig, axes = plt.subplots(2,1)
            # [ax1.add_artist(artist) for artist in anim1[-1]]
            # [axes[1].add_artist(artist) for artist in new_anim1[-1]]

            # Add each polygon to the axis
            # for polygon in anim1[0]:
            #     if type(polygon) == matplotlib.patches.Polygon:
            #         ax.add_patch(polygon)

            # ani = animation.ArtistAnimation(fig, anim1, interval=0.1*1000, blit=True)

            # Show the plot
            plt.show()
            # plt.pause(1)
            break 

            # Get human pref -> TODO
            # pref = TODO
            # preferences.append(pref)
            
            pass
        return np.array(preferences)
    
    def animate(fig, arts_list, sperframe=0.1):
        ani = animation.ArtistAnimation(fig, arts_list, interval=sperframe * 1000, blit=True)
        return ani

    # Define a function for the pause
    # TODO  
