#! /usr/bin/python3
import collections
import tqdm
import statistics
import yaml
from src.a2c_model import *
from src.gym_env import *

num_actions = env.action_space.n  # 2
num_hidden_units = 128

model = ActorCritic(num_actions, num_hidden_units)

huber_loss = tf.keras.losses.Huber(reduction=tf.keras.losses.Reduction.SUM)
optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)


min_episodes_criterion = 100
max_episodes = 10000
max_steps_per_episode = 500

# `CartPole-v1` is considered solved if average reward is >= 475 over 500
# consecutive trials
reward_threshold = 475
running_reward = 0

# The discount factor for future rewards
gamma = 0.99

# Keep the last episodes reward
episodes_reward: collections.deque = collections.deque(maxlen=min_episodes_criterion)

t = tqdm.trange(max_episodes)


def compute_loss(action_probs: tf.Tensor,
                 values: tf.Tensor,
                 returns: tf.Tensor) -> tf.Tensor:
    """Computes the combined Actor-Critic loss."""

    advantage = returns - values

    action_log_probs = tf.math.log(action_probs)
    actor_loss = -tf.math.reduce_sum(action_log_probs * advantage)

    critic_loss = huber_loss(values, returns)

    return actor_loss + critic_loss


@tf.function
def train_step(train_initial_state: tf.Tensor,
               train_model: tf.keras.Model,
               train_optimizer: tf.keras.optimizers.Optimizer,
               train_gamma: float,
               max_steps_per_train_episode: int) -> tf.Tensor:
    """Runs a model training step."""

    with tf.GradientTape() as tape:
        # Run the model for one episode to collect training data
        action_probs, values, rewards = run_episode(
            train_initial_state, train_model, max_steps_per_train_episode)

        # Calculate the expected returns
        returns = get_expected_return(rewards, train_gamma)

        # Convert training data to appropriate TF tensor shapes
        action_probs, values, returns = [
            tf.expand_dims(x, 1) for x in [action_probs, values, returns]]

        # Calculate the loss values to update our network
        loss = compute_loss(action_probs, values, returns)

    # Compute the gradients from the loss
    grads = tape.gradient(loss, train_model.trainable_variables)

    # Apply the gradients to the model's parameters
    train_optimizer.apply_gradients(zip(grads, train_model.trainable_variables))

    train_episode_reward = tf.math.reduce_sum(rewards)

    return train_episode_reward


if __name__ == '__main__':
    i = 0
    for i in t:
        initial_state, info = env.reset()
        env.render()
        initial_state = tf.constant(initial_state, dtype=tf.float32)
        episode_reward = int(train_step(
            initial_state, model, optimizer, gamma, max_steps_per_episode))

        episodes_reward.append(episode_reward)
        running_reward = statistics.mean(episodes_reward)

        t.set_postfix(
            episode_reward=episode_reward, running_reward=running_reward)

        # Show the average episode reward every 10 episodes
        if i % 10 == 0:
            pass  # print(f'Episode {i}: average reward: {avg_reward}')

        if running_reward > reward_threshold and i >= min_episodes_criterion:
            break

    model.save('model/model.keras')

    print(f'\nSolved at episode {i}: average reward: {running_reward:.2f}!')
