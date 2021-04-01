"""Hierarchical Attention Networks for TensorFlow."""

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike

from typeguard import typechecked


def document_preprocessing(
    review, words_maxlen=50, sentences_maxlen=10, tokenizer=tokenizer
):
    """Preprocessing function to build appropriate padded sequences for HAN.

    Parameters
    ----------
    review: list.
        List of sentences (strings) of the review.

    words_maxlen: int.
        Maximal length/number of words for a sentence.

    sentences_maxlen: int.
        Maximal length/number of sentences for a review.

    Returns
    -------
    padded_sequences: tf.Tensor.
        Tensor of shape (sentences_maxlen, words_maxlen)
    """
    sequences = tokenizer.texts_to_sequences(review)
    padded_sequences = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=words_maxlen, padding="post"
    )

    if padded_sequences.shape[0] < sentences_maxlen:
        padded_sequences = tf.pad(
            padded_sequences,
            paddings=tf.constant(
                [[0, sentences_maxlen - padded_sequences.shape[0]], [0, 0]]
            ),
        )
    elif padded_sequences.shape[0] > sentences_maxlen:
        padded_sequences = padded_sequences[:sentences_maxlen]

    assert padded_sequences.shape == (sentences_maxlen, words_maxlen)
    return padded_sequences


# class HierarchicalAttentionNetwork(tf.keras.Model):
#     """Hierarchical Attention Network for TensorFlow.
#
#     Title: Hierarchical Attention Networks for Documents Classification
#     Link: https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf
#
#     Example of usage:
#     ```python
#     import textflow
#
#     han = textflow.han.HierarchicalAttentionNetwork()
#     han.compile(loss="categorical_crossentropy", optimizer="adam")
#     ```
#     """
#
#     def __init__(self):
#         super(HierarchicalAttentionNetwork, self).__init__()
#
#     def call(self, inputs):
#         pass
#
#     @staticmethod
#     def prepare_dataset(self):
#         pass
#
#     def get_config(self):
#         pass


class HierarchicalAttentionNetwork(tf.keras.Model):
    """Hierarchical Attention Network implementation.

    Reference :
    * Hierarchical Attention Networks for Document Classification : https://www.cs.cmu.edu/~./hovy/papers/16HLT-hierarchical-attention-networks.pdf

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
        vocab_size,
        embedding_dim,
        pretrained_weights,
        gru_units,
        attention_units,
        classifier_units,
    ):
        """Skigpram class constructor.

        Parameters
        ----------
        vocab_size: int.
            Size of the vocabulary.

        embedding_dim: int.
            Dimension of trained word2vec Skipgram embeddings.

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
        self.SentenceAttention = Attention(units=attention_units)
        self.DocumentGRU = tf.keras.layers.Bidirectional(
            tf.keras.layers.GRU(
                units=gru_units, activation="tanh", return_sequences=True
            ),
            merge_mode="concat",
        )
        self.DocumentAttention = Attention(units=attention_units)
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

        return


def test():
    han_model = HierarchicalAttentionNetwork(
        vocab_size=4096,
        embedding_dim=128,
        pretrained_weights=None,
        gru_units=32,
        attention_units=32,
        classifier_units=5,
    )
    test_batch = tf.zeros((256, 10, 50))
    test_output = han_model(test_batch)
    assert test_output.shape == (256, 5)
