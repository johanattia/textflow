"Some preprocessing functions."

from typing import Iterable, Optional, Tuple

import numpy as np
import tensorflow as tf

from tqdm.auto import tqdm


# TODO: prepare_skipgram


def prepare_skipgram(
    texts: Iterable[str],
    window_size: int = 4,
    negative_samples: Optional[int] = 2,
    buffer_size: Optional[int] = None,
) -> Tuple[tf.data.Dataset, tf.keras.preprocessing.text.Tokenizer]:
    """Build skipgram dataset and tokenizer from texts. A skipgram dataset is composed
    of skipgram pairs (center_word, context_words). Note that context_words may
    contain both positive and negative examples.

    Args:
        texts (Iterable[str]): A corpus of texts.
        window_size (int, optional): Sliding window for skipgram dataset building.
            Defaults to 4.
        negative_samples (int, optional): Number of negative samples to generate for
            each target_word. Defaults to 2.
        buffer_size (int, optional): Buffer size for tf.data.Dataset shuffling.
            Defaults to None.

    Returns:
        dataset (tf.data.Dataset): Word2vec Skipgram dataset composed pairs
            (center_word, context_words).
        tokenizer (tf.keras.preprocessing.text.Tokenizer): TensorFlow/Keras tokenizer
            built from texts. NB: 0 and 1 are reserved indexes for padding and unknown/oov
            token ('[UNK]') respectively.
    """
    raise NotImplementedError