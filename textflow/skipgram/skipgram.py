"""Skipgram for TensorFlow."""

from typing import Optional
from typeguard import typechecked

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike, Number

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

    w2v = textflow.skipgram.Skipgram(dimension=300, vocab_size=12000)
    dataset = w2v.prepare_dataset(documents)
    w2v.compile(optimizer="adam")

    history = w2v.fit(dataset, epoch=10)
    ```
    """

    @typechecked
    def __init__(
        self,
        vocab_size: Number,
        embedding_dim: Number,
        window_size: Number = 4,
        negative_samples: Number = 1,
        **kwargs
    ):
        """Skigpram class constructor.

        Arguments:
            vocab_size: int.
                Size of the vocabulary.
            embedding_dim: int.
                Dimension of trained word2vec Skipgram embeddings.

        """
        # Inheritance
        super(Skipgram, self).__init__()

        # Skipgram parameters
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.window_size = window_size
        self.negative_samples = negative_samples

        # Layers
        self.target_embedding = tf.keras.layers.Embedding(
            self.vocab_size,
            self.embedding_dim,
            input_length=1,
            name="w2v_embedding",
        )
        self.context_embedding = tf.keras.layers.Embedding(
            self.vocab_size,
            self.embedding_dim,
            input_length=4 + 1,  # number of negative samples = 4
        )
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
        sampling_table: Optional = None,
        shuffle: bool = True,
        categorical: bool = False,
    ) -> tf.data.Dataset:
        pass

    def get_config(self) -> dict:
        config = {
            "embedding_dim": self.embedding_dim,
            "vocab_size": self.vocab_size,
            "window_size": self.window_size,
            "negative_samples": self.negative_samples,
        }
        return config
