#! /usr/bin/python3
from PIL import Image
import gym
import tensorflow as tf
import numpy as np

tf.random.set_seed(42)
np.random.seed(42)
render_env = gym.make("CartPole-v1", render_mode='rgb_array')

max_steps_per_episode = 500


def render_episode(env: gym.Env, render_model: tf.keras.Model, max_steps: int):
    state, info = env.reset()
    state = tf.constant(state, dtype=tf.float32)
    screen = env.render()
    render_images = [Image.fromarray(screen)]

    for i in range(1, max_steps + 1):
        state = tf.expand_dims(state, 0)
        action_probs, _ = render_model(state)
        action = np.argmax(np.squeeze(action_probs))

        state, reward, done, truncated, info = env.step(action)
        state = tf.constant(state, dtype=tf.float32)

        # Render screen every 10 steps
        if i % 10 == 0:
            screen = env.render()
            render_images.append(Image.fromarray(screen))

        if done:
            break

    return render_images


if __name__ == '__main__':
    model = tf.keras.models.load_model('model/model.keras')

    # Save GIF image
    images = render_episode(render_env, model, max_steps_per_episode)
    image_file = 'model/cartpole-v1.gif'
    # loop=0: loop forever, duration=1: play each frame for 1ms
    images[0].save(image_file, save_all=True, append_images=images[1:], loop=0, duration=1)
