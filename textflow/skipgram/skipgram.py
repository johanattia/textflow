"""Skipgram for TensorFlow."""

from typing import Optional
from typeguard import typechecked

import numpy as np
import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike, Number

# TODO: call method, similarity, prepare_dataset, pretrained_weights, tokenizer, fit, get_config...


class Skipgram(tf.keras.Model):
    """Negative Sampling Skipgram for TensorFlow.

    References:
    * Distributed Representations of Words and Phrases and their Compositionality, https://arxiv.org/pdf/1310.4546.pdf
    * Enriching Word Vectors with Subword Information, https://arxiv.org/pdf/1607.04606.pdf
    * Advances in Pre-Training Distributed Word Representations, https://arxiv.org/pdf/1712.09405.pdf

    Example of usage:
    ```python
    import textflow

    dataset = textflow.skipgram.Skipgram.prepare_dataset(documents)

    word2vec = textflow.skipgram.Skipgram(dimension=128, vocab_size=12000)
    word2vec.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False)
    )

    history = word2vec.fit(dataset, epoch=20)
    ```
    """

    @typechecked
    def __init__(
        self,
        vocab_size: int,
        dimension: int,
        tokenizer: Optional[tf.keras.preprocessing.text.Tokenizer],  # ?
        target_weights: Optional[TensorLike],
        context_weights: Optional[TensorLike],
    ):
        """Skigpram class constructor.

        Args:
            vocab_size: int.
                Size of the vocabulary.

            dimension: int.
                Dimension of word2vec Skipgram embeddings.

            tokenizer: tensorflow.keras Tokenizer.
                Tokenizer.

            target_weights: array-like.
                Pretrained weights for skipgram embeddings initialization.

            context_weights: array-like.
                Pretrained weights for context embeddings initialization.
        """
        super(Skipgram, self).__init__()
        self.tokenizer = tokenizer

        # Skipgram parameters
        self.vocab_size = vocab_size
        self.dimension = dimension

        # Skigram (target) Embedding Layer
        if target_weights is not None:
            target_initializer = tf.keras.initializers.Constant(target_weights)
        else:
            target_initializer = "uniform"

        self.target_embedding = tf.keras.layers.Embedding(
            self.vocab_size,
            self.dimension,
            embeddings_initializer=target_initializer,
            trainable=True,
            name="skipgram_vector",
        )

        # Context Embedding Layer
        if context_weights is not None:
            target_initializer = tf.keras.initializers.Constant(target_weights)
        else:
            target_initializer = "uniform"

        self.context_embedding = tf.keras.layers.Embedding(
            self.vocab_size,
            self.dimension,
            embeddings_initializer=tf.keras.initializers.Constant(context_weights),
            trainable=True,
            name="context_vector",
        )

        # Other layers
        self.dots = tf.keras.layers.Dot(axes=(3, 2))
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inputs, training: bool = None, mask: FloatTensorLike = None):
        """Model forward method.

        Arguments:
            inputs: FloatTensorLike. Pair of (center_word, context_words).
                Context words contains both positive and negative examples.
            training: bool. Indicating whether to run the `Network` in training mode or inference mode.
                Useless here.
            mask: FloatTensorLike. Mask or list of masks.
                Useless here.

        Returns:
            FloatTensorLike. Dot products between center word and context words.
        """
        target, context = inputs

        we = self.target_embedding(target)
        ce = self.context_embedding(context)
        dots = self.dots([ce, we])

        return self.flatten(dots)

    def prepare_dataset(
        self,
        sampling_table: Optional[TensorLike],
        window_size: int = 4,
        negative_samples: int = 1,
        shuffle: bool = True,
        categorical: bool = False,
    ) -> tf.data.Dataset:
        pass

    def get_config(self) -> dict:
        config = {
            "dimension": self.dimension,
            "vocab_size": self.vocab_size,
            "tokenizer": self.tokenizer.get_config(),
            "weights": self.weights,
        }
        return config
