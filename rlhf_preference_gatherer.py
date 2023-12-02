import abc
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import matplotlib
from matplotlib.animation import ArtistAnimation, FuncAnimation


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
    def __init__(self, coeff, gamma):
        self.coeff = coeff
        self.gamma = gamma

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
        preferences = []

        for trajectory_pair in trajectory_pairs:
            trajectory1 = trajectory_pair[0].get_transitions()
            trajectory2 = trajectory_pair[1].get_transitions()

            # Calculate reward for both trajectories
            reward1 = self.reward_trajectory(trajectory1)
            reward2 = self.reward_trajectory(trajectory2)

            # Compute preference scores
            if reward1 > reward2:
                preferences.append(1)
            elif reward1 < reward2:
                preferences.append(0)
            else:
                preferences.append(0.5)

        return np.array(preferences)

    def reward_trajectory(self, trajectory):
        """Compute reward for a trajectory."""
        reward = 0
        for i, transition in enumerate(trajectory):
            reward += self.gamma**i * self.reward_transition(transition)
        return reward
    
    def reward_transition(self, transition):
        return np.dot(self.coeff, transition.reward_features)
    

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
        preferences = []

        # TODO: implement human pref

        for trajectory_pair in trajectory_pairs:
            # Retrieve animations
            anim1, fig, ax = trajectory_pair[0].get_animation()
            # frames = anim1._framedata
            print(f"Animation: {type(anim1[0][0])}")

            # Add each polygon to the axis
            # for polygon in anim1[0]:
            #     if type(polygon) == matplotlib.patches.Polygon:
            #         ax.add_patch(polygon)

            ani = animation.ArtistAnimation(fig, anim1, interval=0.1*1000, blit=True)

            # Show the plot
            plt.show()
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
