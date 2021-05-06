"""Hierarchical Attention Networks for TensorFlow."""

from typing import Iterable
from typeguard import typechecked

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike

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
        pretrained_weights: FloatTensorLike,
        gru_units: int,
        attention_units: int,
        classifier_units: int,
    ):
        """Hierarchical Attention Networks constructor.

        Args:
            vocab_size (int): [description]
            embedding_dim (int): [description]
            pretrained_weights (FloatTensorLike): [description]
            gru_units (int): [description]
            attention_units (int): [description]
            classifier_units (int): [description]
        """
        super(HierarchicalAttentionNetwork, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            vocab_size, embedding_dim, weights=[pretrained_weights], trainable=True
        )
        self.SentenceGRU = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units=gru_units, activation="tanh", return_sequences=True
            ),
            merge_mode="concat",
        )
        self.SentenceAttention = AttentionLayer(units=attention_units)
        self.DocumentGRU = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units=gru_units, activation="tanh", return_sequences=True
            ),
            merge_mode="concat",
        )
        self.DocumentAttention = AttentionLayer(units=attention_units)
        self.fc = tf.keras.layers.Dense(units=classifier_units)

    def call(self, x):
        """Model forward method."""
        sentences_vectors, _ = self.sentence_encoder(x)
        sentences_vectors = self.DocumentGRU(sentences_vectors)
        document_vector, _ = self.DocumentAttention(sentences_vectors)

        return self.fc(document_vector)

    def sentence_encoder(self, x):
        """Sentence encoder with Bidirectional GRU and Attention."""
        x = self.embedding(x)
        x = tf.keras.layers.TimeDistributed(self.SentenceGRU)(x)
        context_vector, attention_weights = self.SentenceAttention(x)

        return context_vector, attention_weights

    def document_encoder(self, x, return_weights=True):
        """Document encoder using overall hierarchical architecture."""
        pass
