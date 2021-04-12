"""Skipgram for TensorFlow."""

from typing import Iterable, Optional, Tuple, Union
from typeguard import typechecked

import numpy as np
from annoy import AnnoyIndex

import tensorflow as tf
from tensorflow_addons.utils.types import TensorLike

from tqdm.auto import tqdm


# TODO:
# Annoy index for word/sentence query
# from_config/classmethod
# make_skipgram_dataset


class VocabularyError(LookupError):
    pass


def make_skipgram_dataset(
    texts: Iterable[str],
    window_size: int = 4,
    negative_samples: Optional[int] = 2,
    buffer_size: Optional[int] = None,
) -> Tuple[tf.data.Dataset, tf.keras.preprocessing.text.Tokenizer]:
    """Build skipgram dataset and tokenizer from texts. A skipgram dataset is composed
    of skipgram pairs (center_word, context_words). Note that context_words may
    contain both positive and negative examples.

    Args:
        texts: Iterable[str].
            A corpus of texts.

        window_size: int, default to 4.
            Sliding window for skipgram dataset building.

        negative_samples: Optional[int], default to 1.
            Number of negative samples to generate for each target_word.

        buffer_size: Optional[int], default to None.
            Buffer size for tf.data.Dataset shuffling.

    Returns:
        dataset: tf.data.Dataset.
            Word2vec Skipgram dataset composed pairs (center_word, context_words).

        tokenizer: tf.keras.preprocessing.text.Tokenizer.
            TensorFlow/Keras tokenizer built from texts. NB: 0 and 1 are reserved indexes for
            padding and unknown/oov token ('[UNK]') respectively.
    """
    raise NotImplementedError


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
    word2vec_dataset, tokenizer = textflow.embedding.make_skipgram_dataset(texts)

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
        dimension: int,
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

            skipgram_initializer: tf.keras.initializers.Initializer (str or instance), default to `uniform`.
                Initializer for skipgram embedding layer.

            context_initializer: tf.keras.initializers.Initializer (str or instance), default to `uniform`.
                Initializer for context embedding layer.
        """
        super(Skipgram, self).__init__()
        self.dimension = dimension

        # Tokenizer
        if isinstance(tokenizer, tf.keras.preprocessing.text.Tokenizer):
            self.tokenizer = tokenizer
            self.vocab_size = max(tokenizer.index_word) + 1  # + 1 for padding value 0
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
        self.search_index = False

    def call(self, inputs: Tuple[TensorLike]) -> tf.Tensor:
        """Model forward method.

        Args:
            inputs: tuple of index tensors (center_word, context_words).
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

    def predict_step(self, data: Union[TensorLike, Tuple[TensorLike]]) -> tf.Tensor:
        """Model inference step. Overrides and follows tf.keras.Model predict_step method:
        https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/engine/training.py#L1412-L1434

        Args:
            data: tuple of tensors (center_word, context_words, weights) or just (center) word index tensor.
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

    def word_vectors(self, words: Iterable[str]) -> tf.Tensor:
        """Getting word vector method.

        Args:
            words: Iterable[str].
                Words for which vectors/embeddings are returned.

        Returns:
            tensor: tf.Tensor.
                Skipgram embeddings of words.

        Raises:
            VocabularyError.
                If one or several word from words argument are out-of-vocabulary.
        """
        indexes = tf.convert_to_tensor(
            self.tokenizer.texts_to_sequences(words), dtype=tf.int32
        )
        oov_mask = tf.equal(indexes, tf.ones_like(indexes, dtype=tf.int32))

        if tf.reduce_any(oov_mask):
            oov_words = (
                tf.boolean_mask(tf.expand_dims(tf.constant(words), axis=-1), oov_mask)
                .numpy()
                .tolist()
            )
            raise VocabularyError(f"{oov_words} not in Skipgram vocabulary.")

        return tf.squeeze(self.predict(indexes), axis=1)

    def sentence_vectors(
        self, sentences: Iterable[str], ignore_oov: bool = True
    ) -> tf.Tensor:
        """Getting sentence vector method through word vector averaging.

        Args:
            sentences: Iterable[str].
                Sentences for which vectors/embeddings are returned.

            ignore_oov: bool, default to True.
                Consider a sentence of n tokens composed of n_i in-vocabulary tokens and
                n_o oov tokens (n_i + n_o = n), two cases proposed:
                - ignore_oov=True : word vector averaging will be made by suming the n_i in-vocabulary
                word vectors and dividing by n.
                - ignore_oov=False : word vector averaging will be made by replacing the n_o oov word
                vectors by a special oov vector.

        Returns:
            tensor: tf.Tensor.
                Average skipgram embedding(s) for sentence.
        """
        indexes = tf.ragged.constant(
            self.tokenizer.texts_to_sequences(sentences), dtype=tf.int32
        )

        if ignore_oov:
            oov_mask = tf.equal(indexes, tf.ones_like(indexes, dtype=tf.int32))

            filtered_indexes = tf.ragged.boolean_mask(indexes, ~oov_mask)
            filtered_vectors = self.predict(filtered_indexes)

            return tf.reduce_sum(filtered_vectors, axis=1) / tf.cast(
                tf.expand_dims(indexes.row_lengths(axis=-1), axis=-1), dtype=tf.float32
            )

        return tf.reduce_mean(self.predict(indexes), axis=1)

    def word_similarity(
        self, words1: Iterable[str], words2: Iterable[str]
    ) -> tf.Tensor:
        """Compute cosine similarity for respective pairs of words.
        words1 and words2 arguments are iterable and must have the same length.

        For instance, let's consider words1 = [w11, w12, ..., w1n] and words2 = [w21, w22, ..., w2n],
        this method will compute [cosine_similarity(w11, w21), ..., cosine_similarity(w1n, w2n)].

        Args:
            word1: Iterable[str].
                First words of word pairs.

            word2: Iterable[str].
                Second words of word pairs.

        Returns:
            tensor: tf.Tensor.
                Cosine similarity tensor.
        """
        word_vectors1 = self.word_vectors(words1)
        word_vectors2 = self.word_vectors(words2)

        return -tf.keras.losses.cosine_similarity(word_vectors1, word_vectors2, axis=1)

    def sentence_similarity(
        self,
        sentences1: Iterable[str],
        sentences2: Iterable[str],
        ignore_oov: bool = True,
    ) -> tf.Tensor:
        """Compute cosine similarity for respective pairs of sentences.
        sentences1 and sentences2 arguments are iterable and must have the same length.

        For instance, let's consider sentences1 = [s11, s12, ..., s1n] and sentences2 = [s21, s22, ..., s2n],
        this method will compute [cosine_similarity(s11, s21), ..., cosine_similarity(s1n, s2n)].

        Args:
            sentences1: Iterable[str].
                First sentences of sentence pairs.

            sentences2: Iterable[str].
                Second sentences of sentence pairs.

        Returns:
            tensor: tf.Tensor.
                Cosine similarity tensor.
        """
        sentence_vectors1 = self.sentence_vectors(sentences1, ignore_oov=ignore_oov)
        sentence_vectors2 = self.sentence_vectors(sentences2, ignore_oov=ignore_oov)

        return -tf.keras.losses.cosine_similarity(
            sentence_vectors1, sentence_vectors2, axis=1
        )

    def create_index(
        self, metric: str = "angular", n_trees: int = 100, n_jobs: int = -1
    ):
        """Create Annoy index for approximate vector search.

        Args:
            metric: str, default to `angular`.
                May also be `euclidean`, `manhattan`, `hamming` or `dot`.

            n_trees: int, default to 100.
                Number of trees for the approximation forest. More trees gives
                higher precision when querying.

            n_jobs: int, default to -1.
                Specifies the number of threads used to build the trees.
                n_jobs=-1 uses all available CPU cores.
        """
        if self.search_index:
            raise AttributeError(
                "Already existed search index for approximate vector search."
            )

        search_index = AnnoyIndex(self.dimension, metric=metric)

        for index in tqdm(
            self.tokenizer.index_word, desc="Skipgram Embedding Indexing"
        ):
            embedding = self.skipgram_embedding(
                tf.convert_to_tensor(index, dtype=tf.int32)
            ).numpy()
            search_index.add_item(index, embedding)

        search_index.build(n_trees=n_trees, n_jobs=n_jobs)
        self.search_index = search_index

    def query_index(self):
        """Query previously created Annoy index."""
        # if self.search_index:
        return NotImplemented

    def save_index(self, filename: str):
        """Save previously created Annoy index."""
        # if self.search_index:
        return NotImplemented

    def get_config(self) -> dict:
        """Get model config for serialization.

        Returns:
            config: dict.
                Skipgram config with tokenizer config added.
        """
        tokenizer_config = self.tokenizer.get_config()
        config = {
            "dimension": self.dimension,
            "vocab_size": self.vocab_size,
            "tokenizer": tokenizer_config,
        }
        return config
