"""Utils for models."""

from typing import Iterable

import tensorflow as tf
from tensorflow_addons.utils.types import TensorLike


def document_preprocessing(
    review: Iterable[str],
    tokenizer: tf.keras.preprocessing.text.Tokenizer,
    words_maxlen: int = 50,
    sentences_maxlen: int = 10,
) -> TensorLike:
    """Preprocessing function to build appropriate padded sequences for HAN.

    Args:
        review (Iterable[str]): List of sentences (strings) of the review.
        tokenizer (tf.keras.preprocessing.text.Tokenizer): [description]
        words_maxlen (int, optional): Maximal length/number of words for a sentence.
            Defaults to 50.
        sentences_maxlen (int, optional): Maximal length/number of sentences for a review.
            Defaults to 10.

    Returns:
        TensorLike: `padded_sequences` tensor of shape (sentences_maxlen, words_maxlen).
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