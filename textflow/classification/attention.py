"""Attention Mechanism used in Hierarchical Attention Networks."""

from typing import Tuple
from typeguard import typechecked


import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike


class AttentionLayer(tf.keras.layers.Layer):
    """Attention mechanism used in "Hierarchical Attention Networks for Document Classification"
    paper.

    ```python
    attention_layer = AttentionLayer(units=64)
    ```
    """

    @typechecked
    def __init__(self, units: int = 64):
        """Attention layer constructor.

        Args:
            units (int, optional): [description]. Defaults to 64.
        """
        super(AttentionLayer, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.u = tf.keras.layers.Dense(1, use_bias=False)

    def call(self, sequence: FloatTensorLike) -> Tuple[FloatTensorLike]:
        """Layer forward method.

        Args:
            sequence (FloatTensorLike): [description]

        Returns:
            Tuple[FloatTensorLike]: [description]
        """
        attention_logits = self.u(tf.nn.tanh(self.W(sequence)))
        attention_weights = tf.nn.softmax(attention_logits, axis=-2)

        weighted_vectors = attention_weights * sequence
        context_vector = tf.reduce_sum(weighted_vectors, axis=-2)

        return context_vector, attention_weights
