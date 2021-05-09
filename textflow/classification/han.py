"""Hierarchical Attention Networks for TensorFlow."""

from typing import Callable, Dict, Iterable, List, Tuple, Union
from typeguard import typechecked

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike


class AttentionLayer(tf.keras.layers.Layer):
    """Attention mechanism used in "Hierarchical Attention Networks for Document
    Classification" paper.

    ```python
    attention_layer = AttentionLayer(projection_units=64)
    ```
    """

    @typechecked
    def __init__(
        self,
        projection_units: int,
        kernel_initializer: Union[
            str, tf.keras.initializers.Initializer
        ] = "glorot_uniform",
        bias_initializer: Union[
            str, tf.keras.initializers.Initializer
        ] = "glorot_uniform",
        kernel_regularizer: Union[str, tf.keras.regularizers.Regularizer] = None,
        bias_regularizer: Union[str, tf.keras.regularizers.Regularizer] = None,
        activity_regularizer: Union[str, tf.keras.regularizers.Regularizer] = None,
        kernel_constraint: Union[str, tf.keras.constraints.Constraint] = None,
        bias_constraint: Union[str, tf.keras.constraints.Constraint] = None,
        context_initializer: Union[
            str, tf.keras.initializers.Initializer
        ] = "glorot_uniform",
        context_regularizer: Union[str, tf.keras.regularizers.Regularizer] = None,
    ):
        """Attention layer constructor. For more details about parameters, see:
        * https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense

        Args:
            projection_units (int): Dimensionality of the projection space, before computing
                attention weights.
            kernel_initializer (Union[ str, tf.keras.initializers.Initializer ], optional): Kernel
                initializer of the projection layer. Defaults to "glorot_uniform".
            bias_initializer (Union[str, tf.keras.initializers.Initializer], optional): Bias initializer
                of the projection layer. Defaults to "glorot_uniform".
            kernel_regularizer (Union[str, tf.keras.regularizers.Regularizer], optional): Kernel
                regularizer of the projection layer. Defaults to None.
            bias_regularizer (Union[str, tf.keras.regularizers.Regularizer], optional): Bias regularizer
                of the projection layer. Defaults to None.
            activity_regularizer (Union[str, tf.keras.regularizers.Regularizer], optional): Activity
                regularizer of the projection layer. Defaults to None.
            kernel_constraint (Union[str, tf.keras.constraints.Constraint], optional): Constraint function
                applied to the `kernel` weights matrix of the projection layer. Defaults to None.
            bias_constraint (Union[str, tf.keras.constraints.Constraint], optional): Constraint function
                applied to the bias vector of the projection layer. Defaults to None.
            context_initializer (Union[ str, tf.keras.initializers.Initializer ], optional):. Initializer of
                the learnable context vector. Defaults to "glorot_uniform".
            context_regularizer (Union[str, tf.keras.regularizers.Regularizer], optional): Regularizer of
                the learnable context vector. Defaults to None.
        """
        super(AttentionLayer, self).__init__()
        self.W = tf.keras.layers.Dense(
            units=projection_units,
            activation="tanh",
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
        )
        self.u = tf.keras.layers.Dense(
            units=1,
            use_bias=False,
            kernel_initializer=context_initializer,
            kernel_regularizer=context_regularizer,
        )

    def call(
        self, inputs: FloatTensorLike, mask: bool = False
    ) -> Tuple[FloatTensorLike]:
        """Attention forward method.

        Args:
            inputs (FloatTensorLike): [description]

        Returns:
            Tuple[FloatTensorLike]: [description]
        """
        attention_logits = self.u(self.W(inputs))
        attention_weights = tf.nn.softmax(attention_logits, axis=-2)

        # weighted_vectors = tf.multiply(attention_weights, inputs)
        weighted_vectors = attention_weights * inputs
        context_vector = tf.reduce_sum(weighted_vectors, axis=-2)

        return context_vector, attention_weights


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
            initializer (Union[str, tf.keras.initializers.Initializer], optional): [description].
                Defaults to "uniform".
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
        sentences_tensor, word_attention_weights = self.sentence_encoder(x)
        document_tensor, sentence_attention_weights = self.document_encoder(
            sentences_tensor
        )
        output = {
            "pred_output": self.dense(document_tensor),
            "word_attention_weights": word_attention_weights,
            "sentence_attention_weights": sentence_attention_weights,
        }

        return output

    def train_step(self, data: Iterable[TensorLike]) -> Dict[str, FloatTensorLike]:
        return NotImplemented

    def test_step(self, data: Iterable[TensorLike]) -> Dict[str, FloatTensorLike]:
        return NotImplemented

    def sentence_encoder(self, x: TensorLike) -> FloatTensorLike:
        """Given words from each sentences, encode the contextual representation of the words from
        the sentence with Bidirectional GRU and Attention, and output a sentence_vector.

        Args:
            x (TensorLike): [description]

        Returns:
            FloatTensorLike: [description]
        """
        x = self.embedding(x)
        x = tf.keras.layers.TimeDistributed(self.WordGRU)(x)
        sentences_tensor, attention_weights = self.WordAttention(x)

        return sentences_tensor, attention_weights

    def document_encoder(self, sentences_tensor: FloatTensorLike) -> FloatTensorLike:
        """Given sentences from each review, encode the contextual representation of the sentences with
        Bidirectional GRU and Attention, and output a document vector.

        Args:
            sentences_tensor (FloatTensorLike): [description]

        Returns:
            FloatTensorLike: [description]
        """
        sentences_tensor = self.SentenceGRU(sentences_tensor)
        document_tensor, attention_weights = self.SentenceAttention(sentences_tensor)

        return document_tensor, attention_weights

    @staticmethod
    def document_to_tensor(
        document: str, tokenizer_func: Callable[[Iterable[str], List[int]]]
    ) -> tf.RaggedTensor:
        """Split document (str) into sentences and return ragged tensor of word indexes (int) using
        `tokenizer_func` argument.

        Args:
            document (str): Document to tokenize.
            tokenizer (Callable[[Iterable[str], List[int]]]: tokenizer function. It has to
                map words (str) into word indexes (int).

        Returns:
            tf.RaggedTensor: ragged tensor of  shape (n_sentences, None). `None` stands
                variable sentence length.
        """
        try:
            from nltk.tokenize import sent_tokenize

        except ImportError:
            raise ImportError(
                "Please install nltk. For details, see: https://www.nltk.org/install.html"
            )

        sentences = sent_tokenize(document)
        sequences = tokenizer_func(sentences)

        return tf.ragged.constant(sequences)
