"""Skipgram for TensorFlow."""

from typing import Iterable, Optional, Tuple, Union
from typeguard import typechecked

import numpy as np
import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike, Number

# TODO: call method, word_similarity, sentence_similarity, prepare_dataset, document_vector, word_vector, tokenizer, get_config...


class Skipgram(tf.keras.Model):
    """Negative Sampling Skipgram for TensorFlow.

    Main reference:
    * Distributed Representations of Words and Phrases and their Compositionality, https://arxiv.org/pdf/1310.4546.pdf

    Other references:
    * Enriching Word Vectors with Subword Information, https://arxiv.org/pdf/1607.04606.pdf
    * Advances in Pre-Training Distributed Word Representations, https://arxiv.org/pdf/1712.09405.pdf

    Example of usage:
    ```python
    import tensorflow as tf
    import textflow

    dimension = 128
    word2vec_dataset, tokenizer = textflow.skipgram.Skipgram.prepare_dataset(texts)

    word2vec = textflow.skipgram.Skipgram(dimension=dimension, tokenizer=tokenizer)
    word2vec.compile(
        optimizer="adam",
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=True)
    )

    history = word2vec.fit(word2vec_dataset, epoch=20)
    ```
    """

    @typechecked
    def __init__(
        self,
        dimension: Number,
        tokenizer: tf.keras.preprocessing.text.Tokenizer,
        target_initializer: Union[str, tf.keras.initializers.Initializer] = "uniform",
        context_initializer: Union[str, tf.keras.initializers.Initializer] = "uniform",
    ):
        """Skigpram class constructor.

        Args:
            dimension: int.
                Dimension of word2vec Skipgram embeddings.

            tokenizer: tf.keras.preprocessing.text.Tokenizer.
                TensorFlow/Keras tokenizer built from a corpus of texts. Note that each skipgram
                dataset used for training - in fit() method for instance - must be generated from this
                tokenizer/text encoder.

            target_initializer: tf.keras.initializers.Initializer (str or instance).
                Initializer for skipgram embedding layer.

            context_initializer: tf.keras.initializers.Initializer (str or instance).
                Initializer for context embedding layer.
        """
        super(Skipgram, self).__init__()
        self.dimension = dimension

        # Tokenizer
        if isinstance(tokenizer, tf.keras.preprocessing.text.Tokenizer):
            self.tokenizer = tokenizer
            self.vocab_size = max(tokenizer.index_word) + 1
        else:
            raise TypeError(
                "tokenizer must be a tf.keras.preprocessing.text.Tokenizer instance."
            )

        # Skigram Embedding Layer
        self.skipgram_embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.dimension,
            embeddings_initializer=target_initializer,
            trainable=True,
            name="skipgram_embedding",
        )

        # Context Embedding Layer
        self.context_embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.dimension,
            embeddings_initializer=context_initializer,
            trainable=True,
            name="context_embedding",
        )

    def call(self, inputs: Tuple[TensorLike, TensorLike]) -> tf.Tensor:
        """Model forward method.

        Args:
            inputs: pair of int tensors-like (center_word, context_words).
                Context words contains both positive and negative examples.

        Returns:
            Dot products: tf.Tensor.
                Dot products between center word and context words. These products are
                logits for entropy loss function.
        """
        target, context = inputs

        target_vectors = self.skipgram_embedding(target)
        context_vectors = self.context_embedding(context)

        target_vectors = tf.expand_dims(target_vectors, axis=2)
        products = tf.squeeze(tf.matmul(context_vectors, target_vectors))

        return products  # tf.nn.softmax/sigmoid/sigmoid_cross_entropy_with_logits

    @staticmethod
    def prepare_dataset(
        texts: Iterable[str],
        window_size: int = 4,
        negative_samples: int = 1,
        buffer: int = 1000,
        shuffle: bool = True,
    ) -> Tuple[tf.data.Dataset, tf.keras.preprocessing.text.Tokenizer]:
        """Build skipgram dataset and tokenizer from texts. A skipgram dataset is composed
        of skipgram pairs (center_word, context_words). Note that context_words may
        contain both positive and negative examples.

        Args:
            texts: Iterable[str].
                A corpus of texts.

            window_size: int.
                Default to 4. Sliding window for skipgram dataset building.

            negative_samples: int.
                Default to 1. Number of negative samples to generate for each target_word.

            suffle: bool.
                Default to True. Whether suffle the output word2vec tf.data.Dataset.

        Returns:
            dataset: tf.data.Dataset.
                Word2vec Skipgram dataset composed pairs (center_word, context_words).

            tokenizer: tf.keras.preprocessing.text.Tokenizer.
                TensorFlow/Keras tokenizer built from texts. NB: 0 and 1 are reserved indexes for
                padding and unknown/oov token ('[UNK]') respectively.
        """
        raise NotImplementedError

    def get_config(self) -> dict:
        config = {
            "dimension": self.dimension,
            "vocab_size": self.vocab_size,
            "tokenizer": self.tokenizer.get_config(),
        }
        return config
