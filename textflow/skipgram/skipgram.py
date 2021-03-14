"""Skipgram for TensorFlow."""

from typing import Optional, Union
from typeguard import typechecked

import numpy as np
import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike, Number

# TODO: call method, similarity, prepare_dataset, pretrained_weights, tokenizer, fit, get_config...


class Skipgram(tf.keras.Model):
    """Negative Sampling Skipgram for TensorFlow.

    Main reference:
    * Distributed Representations of Words and Phrases and their Compositionality, https://arxiv.org/pdf/1310.4546.pdf

    Other references:
    * Enriching Word Vectors with Subword Information, https://arxiv.org/pdf/1607.04606.pdf
    * Advances in Pre-Training Distributed Word Representations, https://arxiv.org/pdf/1712.09405.pdf

    Example of usage:
    ```python
    import textflow

    dataset = textflow.skipgram.Skipgram.prepare_dataset(documents)

    word2vec = textflow.skipgram.Skipgram(vocab_size=12000, dimension=128)
    word2vec.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    )

    history = word2vec.fit(dataset, epoch=20)
    ```
    """

    @typechecked
    def __init__(
        self,
        vocab_size: Number,
        dimension: Number,
        tokenizer: Optional[tf.keras.preprocessing.text.Tokenizer] = None,
        target_initializer: Union[str, tf.keras.initializers.Initializer] = "uniform",
        context_initializer: Union[str, tf.keras.initializers.Initializer] = "uniform",
    ):
        """Skigpram class constructor.

        Args:
            vocab_size: int.
                Size of the vocabulary.

            dimension: int.
                Dimension of word2vec Skipgram embeddings.

            tokenizer: tensorflow.keras Tokenizer.
                Tokenizer.

            target_initializer: tf.keras.initializers.Initializer (str or instance).
                Initializer for skipgram embedding layer.

            context_initializer: tf.keras.initializers.Initializer (str or instance).
                Initializer for context embedding layer.
        """
        super(Skipgram, self).__init__()

        # Skipgram parameters
        self.vocab_size = vocab_size
        self.dimension = dimension

        # Tokenizer
        if isinstance(tokenizer, tf.keras.preprocessing.text.Tokenizer):
            self.tokenizer = tokenizer
        else:
            raise TypeError(
                "tokenizer must be a tf.keras.preprocessing.text.Tokenizer instance."
            )

        # Skigram (target) Embedding Layer
        if isinstance(target_initializer, (str, tf.keras.initializers.Initializer)):
            self.target_embedding = tf.keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.dimension,
                embeddings_initializer=target_initializer,
                trainable=True,
                name="target_vector",
            )
        else:
            raise TypeError(
                "target_initializer must be an initializer identifier (str) or a tf.keras.initializers.Initializer instance"
            )

        # Context Embedding Layer
        if isinstance(context_initializer, (str, tf.keras.initializers.Initializer)):
            self.context_embedding = tf.keras.layers.Embedding(
                input_dim=self.vocab_size,
                output_dim=self.dimension,
                embeddings_initializer=context_initializer,
                trainable=True,
                name="context_vector",
            )
        else:
            raise TypeError(
                "context_initializer must be an initializer identifier (str) or a tf.keras.initializers.Initializer instance"
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
