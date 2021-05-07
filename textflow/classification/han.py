"""Hierarchical Attention Networks for TensorFlow."""

from typing import Union
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
        vocab_size=max(tokenizer.index_word.keys())+1,
        embedding_dim=128,
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

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        gru_units: int,
        attention_units: int,
        classifier_units: int,
        initializer: Union[str, tf.keras.initializers.Initializer] = "uniform",
    ):
        """Hierarchical Attention Network class constructor.

        Args:
            vocab_size (int): [description]
            embedding_dim (int): [description]
            gru_units (int): [description]
            attention_units (int): [description]
            classifier_units (int): [description]
            initializer (Union[str, tf.keras.initializers.Initializer], optional): [description]. Defaults to "uniform".
        """
        super(HierarchicalAttentionNetwork, self).__init__()

        self.embedding = tf.keras.layers.Embedding(
            vocab_size,
            embedding_dim,
            embeddings_initializer=initializer,
            trainable=True,
        )
        self.WordGRU = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units=gru_units, activation="tanh", return_sequences=True
            ),
            merge_mode="concat",
        )
        self.WordAttention = AttentionLayer(units=attention_units)
        self.SentenceGRU = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units=gru_units, activation="tanh", return_sequences=True
            ),
            merge_mode="concat",
        )
        self.SentenceAttention = AttentionLayer(units=attention_units)
        self.fc = tf.keras.layers.Dense(units=classifier_units)

    def call(self, x: TensorLike) -> FloatTensorLike:
        """Model forward method.

        Args:
            x (TensorLike): [description]

        Returns:
            FloatTensorLike: [description]
        """
        sentences_vectors, _ = self.word_to_sentence_encoder(x)
        document_vector, _ = self.sentence_to_document_encoder(sentences_vectors)
        return self.fc(document_vector)

    def word_to_sentence_encoder(self, x: TensorLike) -> FloatTensorLike:
        """Given words from each sentences, encode the contextual representation of
        the words from the sentence with Bidirectional GRU and Attention, and output
        a sentence_vector.

        Args:
            x (TensorLike): [description]

        Returns:
            FloatTensorLike: [description]
        """
        x = self.embedding(x)
        x = tf.keras.layers.TimeDistributed(self.WordGRU)(x)
        context_vector, attention_weights = self.WordAttention(x)

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
        sentences_vectors = self.SentenceGRU(sentences_vectors)
        document_vector, attention_weights = self.SentenceAttention(sentences_vectors)

        return document_vector, attention_weights
