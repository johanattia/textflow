"""Hierarchical Attention Networks for TensorFlow."""

from typing import Callable, Iterable, Union
from typeguard import typechecked

import numpy as np
import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike

from .attention import AttentionLayer


class HierarchicalAttentionNetwork(tf.keras.Model):
    """Hierarchical Attention Network implementation.

    Reference :
    * https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf

    ```python
    han_model = HierarchicalAttentionNetwork(
        vocabulary_size=max(tokenizer.index_word.keys())+1,
        embed_dimension=128,
        pretrained_weights=pretrained_weights,
        gru_units=32,
        attention_units=32,
        classifier_units=5
    )
    han_model.compile(
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"]
    )
    ```
    """

    @typechecked
    def __init__(
        self,
        vocabulary_size: int,
        n_classes: int,
        attention_units: int = 100,
        word_recurrent_units: int = 100,
        sentence_recurrent_units: int = 100,
        embed_dimension: int = 200,
        initializer: Union[str, tf.keras.initializers.Initializer] = "uniform",
        **kwargs
    ):
        """Hierarchical Attention Network class constructor.

        Args:
            vocabulary_size (int): [description]
            n_classes (int): [description]
            attention_units (int, optional): [description]. Defaults to 100.
            word_recurrent_units (int, optional): [description]. Defaults to 100.
            sentence_recurrent_units (int, optional): [description]. Defaults to 100.
            embed_dimension (int, optional): [description]. Defaults to 200.
            initializer (Union[str, tf.keras.initializers.Initializer], optional): [description]. Defaults to "uniform".
        """
        super(HierarchicalAttentionNetwork, self).__init__(**kwargs)

        self.embedding = tf.keras.layers.Embedding(
            vocabulary_size,
            embed_dimension,
            embeddings_initializer=initializer,
            trainable=True,
        )
        self.WordGRU = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units=word_recurrent_units, activation="tanh", return_sequences=True
            ),
            merge_mode="concat",
        )
        self.WordAttention = AttentionLayer(units=attention_units)
        self.SentenceGRU = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units=sentence_recurrent_units, activation="tanh", return_sequences=True
            ),
            merge_mode="concat",
        )
        self.SentenceAttention = AttentionLayer(units=attention_units)
        self.dense = tf.keras.layers.Dense(units=n_classes)

    def call(
        self, x: TensorLike
    ) -> FloatTensorLike:  # [batch_size, n_sentence, n_word_idx]
        """Model forward method.

        Args:
            x (TensorLike): [description]

        Returns:
            FloatTensorLike: [description]
        """
        sentences_vectors, _ = self.word_to_sentence_encoder(x)
        document_vector, _ = self.sentence_to_document_encoder(sentences_vectors)
        return self.dense(document_vector)  # [batch_size, n_classes]

    def word_to_sentence_encoder(self, x: TensorLike) -> FloatTensorLike:
        """Given words from each sentences, encode the contextual representation of
        the words from the sentence with Bidirectional GRU and Attention, and output
        a sentence_vector.

        Args:
            x (TensorLike): [description]

        Returns:
            FloatTensorLike: [description]
        """
        x = self.embedding(x)  # [batch_size, n_sentence, n_word_idx, embedding_dim]
        x = tf.keras.layers.TimeDistributed(self.WordGRU)(
            x
        )  # [batch_size, n_sentence, n_word_idx, embedding_dim]
        context_vector, attention_weights = self.WordAttention(
            x
        )  # [batch_size, n_sentence, embedding_dim]

        return context_vector, attention_weights

    def sentence_to_document_encoder(
        self, sentences_vectors: FloatTensorLike
    ) -> FloatTensorLike:
        """Given sentences from each review, encode the contextual representation of
        the sentences with Bidirectional GRU and Attention, and output
        a document vector.

        Args:
            sentences_vectors (FloatTensorLike): [description]

        Returns:
            FloatTensorLike: [description]
        """
        sentences_vectors = self.SentenceGRU(
            sentences_vectors
        )  # [batch_size, n_sentence, embedding_dim]
        document_vector, attention_weights = self.SentenceAttention(
            sentences_vectors
        )  # [batch_size, projection_dim]

        return document_vector, attention_weights

    @staticmethod
    def sentences_to_tensor(
        sentences: Iterable[str], tokenizer: tf.keras.preprocessing.text.Tokenizer
    ) -> tf.RaggedTensor:
        """[summary]

        Args:
            sentences (Iterable[str]): [description]
            tokenizer (tf.keras.preprocessing.text.Tokenizer): [description]

        Returns:
            tf.RaggedTensor: [description]
        """
        sequences = tokenizer.texts_to_sequences(sentences)
        return tf.ragged.constant(sequences)
