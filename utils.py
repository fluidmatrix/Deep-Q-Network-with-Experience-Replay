import numpy as np
import tensorflow as tf
import random
import matplotlib.pyplot as plt

def get_action(q_values, epsilon):
    """
    Selects an action using the epsilon-greedy policy.

    Args:
        q_values (tf.Tensor): Q-values predicted by the model for the current state.
        epsilon (float): Exploration rate (probability of random action).

    Returns:
        int: Selected action.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(q_values.shape[-1])
    else:
        return int(np.argmax(q_values))

def check_update_conditions(timestep, update_frequency, memory_buffer, batch_size=64):
    """
    Determines if the agent should perform a learning update.

    Args:
        timestep (int): Current time step in the episode.
        update_frequency (int): Number of steps between updates.
        memory_buffer (deque): Experience replay buffer.

    Returns:
        bool: True if an update should be performed, False otherwise.
    """
    return (timestep % update_frequency == 0) and (len(memory_buffer) >= batch_size)

def get_experiences(memory_buffer, batch_size=64):
    """
    Samples a batch of experiences from the memory buffer.

    Args:
        memory_buffer (deque): Experience replay buffer.
        batch_size (int): Number of experiences to sample.

    Returns:
        Tuple of (states, actions, rewards, next_states, done_flags)
    """
    experiences = random.sample(memory_buffer, k=batch_size)

    states = np.array([e.state for e in experiences], dtype=np.float32)
    actions = np.array([e.action for e in experiences], dtype=np.int32)
    rewards = np.array([e.reward for e in experiences], dtype=np.float32)
    next_states = np.array([e.next_state for e in experiences], dtype=np.float32)
    done_vals = np.array([float(e.done) for e in experiences], dtype=np.float32)

    return (states, actions, rewards, next_states, done_vals)

def update_target_network(q_network, target_q_network):
    """
    Copies weights from the Q-network to the target Q-network.

    Args:
        q_network (tf.keras.Model): Main Q-network.
        target_q_network (tf.keras.Model): Target Q-network.
    """
    target_q_network.set_weights(q_network.get_weights())

def get_new_eps(epsilon, min_epsilon=0.01, decay=0.995):
    """
    Decays the exploration rate ε after each episode.

    Args:
        epsilon (float): Current exploration rate.
        min_epsilon (float): Minimum exploration rate.
        decay (float): Decay rate per episode.

    Returns:
        float: Updated epsilon value.
    """
    return max(min_epsilon, epsilon * decay)

def plot_history(total_point_history, smoothing_window=20):
    """
    Plots the total points per episode and a smoothed average.

    Args:
        total_point_history (list): Total reward per episode.
        smoothing_window (int): Number of episodes to average over for smoothing.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(total_point_history, label='Episode Reward')
    
    if len(total_point_history) >= smoothing_window:
        moving_avg = np.convolve(
            total_point_history, 
            np.ones(smoothing_window)/smoothing_window, 
            mode='valid'
        )
        plt.plot(range(smoothing_window - 1, len(total_point_history)), moving_avg, label=f'{smoothing_window}-Episode Moving Avg', color='orange')

    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.title('Training Progress')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()
