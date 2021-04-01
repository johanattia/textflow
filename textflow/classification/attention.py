"""Attention Mechanism used in Hierarchical Attention Networks."""

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike

from typeguard import typechecked


class Attention(tf.keras.layers.Layer):
    """Attention mechanism used in "Hierarchical Attention Networks for Document Classification" paper.

    ```python
    attention_layer = Attention(units=64)
    ```
    """

    def __init__(self, units):
        """Attention layer constructor.

        Parameters
        ----------
        units: int.
            Dimension of the projection space.
        """
        super(Attention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.u = tf.keras.layers.Dense(1)

    def call(self, sequence):
        """Layer forward method."""
        attention_logits = self.u(tf.nn.tanh(self.W(sequence)))
        attention_weights = tf.nn.softmax(attention_logits)

        weighted_vectors = attention_weights * sequence
        context_vector = tf.reduce_sum(weighted_vectors, axis=-2)

        return context_vector, attention_weights


def test():
    batch1 = tf.random.normal((16, 10, 50, 128))
    batch2 = tf.random.normal((16, 10, 128))

    attention = Attention(units=64)

    (att_batch1, weights_batch1), (att_batch2, weights_batch2) = (
        attention(batch1),
        attention(batch2),
    )
    assert att_batch1.shape == (16, 10, 128) & weights_batch1.shape == (16, 10, 50, 1)
    assert att_batch2.shape == (16, 128) & weights_batch2.shape(16, 10, 1)
