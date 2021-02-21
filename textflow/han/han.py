"""Hierarchical Attention Networks for TensorFlow."""

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike

from typeguard import typechecked


class HierarchicalAttentionNetwork(tf.keras.Model):
    """Hierarchical Attention Network for TensorFlow.

    Title: Hierarchical Attention Networks for Documents Classification
    Link: https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf

    Example of usage:
    ```python
    import textflow

    han = textflow.han.HierarchicalAttentionNetwork()
    han.compile(loss="categorical_crossentropy", optimizer="adam")
    ```
    """

    def __init__(self):
        super(HierarchicalAttentionNetwork, self).__init__()

    def call(self, inputs):
        pass

    @staticmethod
    def prepare_dataset(self):
        pass

    def get_config(self):
        pass
