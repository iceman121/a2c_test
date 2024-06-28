import tensorflow as tf
from tensorflow.keras import layers
from typing import Tuple

tf.random.set_seed(42)


class ActorCritic(tf.keras.Model):
    """Combined actor-critic network."""
    def __init__(
            self,
            num_actions: int,
            num_hidden_units: int,
            **kwargs):
        """Initialize."""
        super().__init__()
        self.num_actions = num_actions
        self.num_hidden_units = num_hidden_units

        self.common = layers.Dense(self.num_hidden_units, activation="relu")
        self.actor = layers.Dense(self.num_actions)
        self.critic = layers.Dense(1)

    def call(self, inputs: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
        x = self.common(inputs)
        return self.actor(x), self.critic(x)

    def get_config(self):
        config = super().get_config()
        config.update({
            'num_actions': self.num_actions,
            'num_hidden_units': self.num_hidden_units,
            'common': self.common,
            'actor': self.actor,
            'critic': self.critic,
        })
        return config

    @classmethod
    def from_config(cls, config):
        config['num_actions'] = tf.keras.utils.deserialize_keras_object(config['num_actions'])
        config['num_hidden_units'] = tf.keras.utils.deserialize_keras_object(config['num_hidden_units'])
        config['common'] = tf.keras.utils.deserialize_keras_object(config['common'])
        config['actor'] = tf.keras.utils.deserialize_keras_object(config['actor'])
        config['critic'] = tf.keras.utils.deserialize_keras_object(config['critic'])
        return cls(**config)
