"""Skipgram for TensorFlow."""

from typing import Iterable, Tuple, Union  # Optional
from typeguard import typechecked

import annoy
import numpy as np

import tensorflow as tf
from tensorflow_addons.utils.types import TensorLike, Number  # FloatTensorLike


# TODO:
# tf.keras.layers.experimental.preprocessing.TextVectorization instead of tokenizer ?
# Annoy index for word/sentence query
# prepare_dataset


class Skipgram(tf.keras.Model):
    """Negative Sampling Skipgram for TensorFlow.

    Main references:
    * Efficient Estimation of Word Representations in Vector Space, https://arxiv.org/pdf/1301.3781.pdf
    * Distributed Representations of Words and Phrases and their Compositionality, https://arxiv.org/pdf/1310.4546.pdf

    Other references:
    * Enriching Word Vectors with Subword Information, https://arxiv.org/pdf/1607.04606.pdf
    * Advances in Pre-Training Distributed Word Representations, https://arxiv.org/pdf/1712.09405.pdf

    Example of usage:
    ```python
    import tensorflow as tf
    import textflow

    dimension = 128
    word2vec_dataset, tokenizer = textflow.embedding.Skipgram.prepare_dataset(texts)

    word2vec = textflow.embedding.Skipgram(dimension=dimension, tokenizer=tokenizer)
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
        skipgram_initializer: Union[str, tf.keras.initializers.Initializer] = "uniform",
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

            skipgram_initializer: tf.keras.initializers.Initializer (str or instance).
                Initializer for skipgram embedding layer.

            context_initializer: tf.keras.initializers.Initializer (str or instance).
                Initializer for context embedding layer.
        """
        super(Skipgram, self).__init__()
        self.dimension = dimension

        # Tokenizer
        if isinstance(tokenizer, tf.keras.preprocessing.text.Tokenizer):
            self.tokenizer = tokenizer
            self.vocab_size = max(tokenizer.index_word) + 1  # for padding
        else:
            raise TypeError(
                "tokenizer must be a tf.keras.preprocessing.text.Tokenizer instance."
            )

        # Skigram Embedding Layer
        self.skipgram_embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.dimension,
            embeddings_initializer=skipgram_initializer,
            mask_zero=True,
            trainable=True,
            name="skipgram_embedding",
        )

        # Context Embedding Layer
        self.context_embedding = tf.keras.layers.Embedding(
            input_dim=self.vocab_size,
            output_dim=self.dimension,
            embeddings_initializer=context_initializer,
            mask_zero=True,
            trainable=True,
            name="context_embedding",
        )

        # Indexing for search
        self.search_index_ = False

    def call(self, inputs: Tuple[TensorLike]) -> tf.Tensor:
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

    def predict_step(self, data: Union[TensorLike, Iterable[TensorLike]]) -> tf.Tensor:
        """Model inference step. Overrides and follows tf.keras.Model predict_step method:
        https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/engine/training.py#L1412-L1434

        Args:
            data: tuple of int tensors-like (center_word, context_words).
                Context words contains both positive and negative examples.

        Returns:
            Dot products: tf.Tensor.
                Dot products between center word and context words. These products are
                logits for entropy loss function.
        """

        def _expand_single_1d_tensor(tensor):
            if (
                isinstance(tensor, tf.Tensor)
                and isinstance(tensor.shape, tf.TensorShape)
                and tensor.shape.rank == 1
            ):
                return tf.expand_dims(tensor, axis=-1)
            return tensor

        data = tf.nest.map_structure(_expand_single_1d_tensor, data)
        x, _, _ = tf.keras.utils.unpack_x_y_sample_weight(data)

        return self.skipgram_embedding(x)

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

    def word_vector(self, word: Iterable[str]) -> tf.Tensor:
        """Getting word vector method.

        Args:
            word: Iterable[str].
                Word(s) for which vector(s)/embedding(s) is/are returned.

        Returns:
            tensor: tf.Tensor.
                Skipgram embedding(s) of word.
        """
        index = tf.convert_to_tensor(
            self.tokenizer.texts_to_sequences(word), dtype=tf.int32
        )
        word_vector = self.predict(index)
        return word_vector

    def sentence_vector(self, sentence: Iterable[str]) -> tf.Tensor:
        """Getting sentence vector method through word vector averaging.

        Args:
            sentence: Iterable[str].
                Sentence(s) for which vector(s)/embedding(s) is/are returned.

        Returns:
            tensor: tf.Tensor.
                Average skipgram embedding(s) for each sentence.
        """
        indexes = tf.ragged.constant(
            self.tokenizer.texts_to_sequences(sentence), dtype=tf.int32
        )
        sentence_vector = tf.reduce_mean(self.predict(indexes), axis=1)
        return sentence_vector

    def word_similarity(self):
        raise NotImplementedError

    def sentence_similarity(self):
        raise NotImplementedError

    def create_index(self):
        raise NotImplementedError

    def query_index(self):
        raise NotImplementedError

    def get_config(self) -> dict:
        config = {
            "dimension": self.dimension,
            "vocab_size": self.vocab_size,
            "tokenizer": self.tokenizer.get_config(),
        }
        return config
