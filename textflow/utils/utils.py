"Utilities functions."

from typing import Iterable, Union

import tensorflow as tf
from tensorflow_addons.utils.types import TensorLike


class VocabularyError(LookupError):
    pass


def expand_1d(data: Iterable[TensorLike]) -> Iterable[TensorLike]:
    """Utility function when overrides tf.keras.Model train_step, test_step
    or predict_step methods. Taken from:
    * https://github.com/tensorflow/tensorflow/blob/v2.4.1/tensorflow/python/keras/engine/data_adapter.py#L1320-L1330

    Args:
        data (Iterable[TensorLike]): tensor(s).

    Returns:
        Iterable[TensorLike]: expanded tensor(s).
    """

    def _expand_single_1d_tensor(tensor):
        if (
            isinstance(tensor, tf.Tensor)
            and isinstance(tensor.shape, tf.TensorShape)
            and tensor.shape.rank == 1
        ):
            return tf.expand_dims(tensor, axis=-1)
        return tensor

    return tf.nest.map_structure(_expand_single_1d_tensor, data)