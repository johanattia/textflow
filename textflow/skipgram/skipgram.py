"""Skipgram for TensorFlow."""

from typing import Optional
from typeguard import typechecked

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike, Number

# TODO: train_step, test_step, inference...


class Skipgram(tf.keras.Model):
    """Negative Sampling Skipgram for TensorFlow.

    References:
    * Distributed Representations of Words and Phrases and their Compositionality, https://arxiv.org/pdf/1310.4546.pdf
    * Enriching Word Vectors with Subword Information, https://arxiv.org/pdf/1607.04606.pdf
    * Advances in Pre-Training Distributed Word Representations, https://arxiv.org/pdf/1712.09405.pdf

    Example of usage:
    ```python
    import textflow

    skipgram = textflow.skipgram.Skipgram(dimension=300, vocab_size=12000)
    dataset = skipgram.prepare_dataset(documents)
    skipgram.compile(optimizer="adam")

    history = skipgram.fit(dataset, epoch=10)
    ```
    """

    @typechecked
    def __init__(
        self,
        dimension: Number = 300,
        vocab_size: Number = None,
        window_size: Number = 4,
        negative_samples: Number = 1,
        **kwargs
    ):
        super(Skipgram, self).__init__(**kwargs)
        self.dimension = dimension
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.negative_samples = negative_samples

        self.target_embedding = tf.keras.layers.Embedding(
            self.vocab_size, self.dimension, name="skipgram_embedding"
        )
        self.context_embedding = tf.keras.layers.Embedding(
            self.vocab_size, self.dimension, name="context_embedding"
        )
        self.dots = tf.keras.layers.Dot()

    def call(self, inputs: FloatTensorLike) -> FloatTensorLike:
        target_word, context_word = inputs
        target_vector = self.target_embedding(target_word)
        context_vector = self.context_embedding(context_word)
        dot_product = tf.math.reduce_sum(
            tf.multiply(target_vector, context_vector), axis=1
        )
        return tf.nn.sigmoid(dot_product)  # loss : from_logits=False

    def prepare_dataset(
        self,
        sampling_table: Optional = None,
        shuffle: bool = True,
        categorical: bool = False,
    ) -> tf.data.Dataset:
        pass

    def get_config(self) -> dict:
        config = {
            "dimension": self.dimension,
            "vocab_size": self.vocab_size,
            "window_size": self.window_size,
            "negative_samples": self.negative_samples,
        }
        return config
