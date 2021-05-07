"""Hierarchical Attention Networks for TensorFlow."""

from typing import Dict, Iterable, Union
from typeguard import typechecked

import numpy as np
import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike

from .attention import AttentionLayer


class HierarchicalAttentionNetwork(tf.keras.Model):
    """Hierarchical Attention Network model.

    Reference :
    * https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf

    ```python
    han_model = HierarchicalAttentionNetwork(
        vocabulary_size=max(tokenizer.index_word.keys())+1,
        n_classes=4
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

    def call(self, x: TensorLike) -> Dict[str, FloatTensorLike]:
        """Model forward method.

        Args:
            x (TensorLike): [description]

        Returns:
            Dict[str, FloatTensorLike]: [description]
        """
        sentences_vectors, word_attention_weights = self.word_to_sentence_encoder(x)
        document_vector, sentence_attention_weights = self.sentence_to_document_encoder(
            sentences_vectors
        )
        output = {
            "pred_output": self.dense(document_vector),
            "word_attention_weights": word_attention_weights,
            "sentence_attention_weights": sentence_attention_weights,
        }
        return output

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

    # def train_step(self,):
    # def test_step(self,):
    # def predict_step(self,):

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
