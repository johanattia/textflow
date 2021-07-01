"""Hierarchical Attention Networks for TensorFlow."""


from typing import Callable, Dict, Iterable, List, Tuple, Union
from typeguard import typechecked

import tensorflow as tf
from tensorflow_addons.utils.types import FloatTensorLike, TensorLike


# TODO:
# AttentionLayer: call method


class AttentionLayer(tf.keras.layers.Layer):
    """Attention mechanism used in "Hierarchical Attention Networks for Document
    Classification" paper.

    ```python
    attention_layer = AttentionLayer(projection_units=64)
    ```

    Args:
        projection_units (int): Dimensionality of the projection space, before
            computing attention weights.
        kernel_initializer (Union[str, Callable], optional): Kernel initializer
            of the projection layer. Defaults to "glorot_uniform".
        kernel_regularizer (Union[str, Callable], optional): Kernel regularizer
            of the projection layer. Defaults to None.
        kernel_constraint (Union[str, Callable], optional): Constraint function
            applied to the `kernel` weights matrix of the projection layer.
            Defaults to None.
        bias_initializer (Union[str, Callable], optional): Bias initializer of
            the projection layer. Defaults to "zeros".
        bias_regularizer (Union[str, Callable], optional): Bias regularizer of
            the projection layer. Defaults to None.
        bias_constraint (Union[str, Callable], optional): Constraint function
            applied to the bias vector of the projection layer. Defaults to None.
        activity_regularizer (Union[str, Callable], optional): Activity regularizer
            of the projection layer. Defaults to None.
        context_initializer (Union[str, Callable], optional): Initializer of the
            learnable context vector. Defaults to "glorot_uniform".
        context_regularizer (Union[str, Callable], optional): Regularizer of the
            learnable context vector. Defaults to None.

    Call Args:
        inputs (FloatTensorLike): Sequence of vectors.
        training (bool, optional): Whether the layer should behave in training
            mode or in inference mode for the dropout layer. Defaults to False.
        mask (bool, optional): Binary (padding) mask of the input sequence. Used
            for attention scores calculation. Defaults to None.
        return_attention_scores (bool, optional): Whether return attention scores.
            Defaults to True.

    Returns:
        context_vector (FloatTensorLike): [description].
        attention_scores (FloatTensorLike): [description].
    """

    @typechecked
    def __init__(
        self,
        projection_units: int,
        dropout: float,
        kernel_initializer: Union[str, Callable] = "glorot_uniform",
        kernel_regularizer: Union[str, Callable] = None,
        kernel_constraint: Union[str, Callable] = None,
        bias_initializer: Union[str, Callable] = "zeros",
        bias_regularizer: Union[str, Callable] = None,
        bias_constraint: Union[str, Callable] = None,
        activity_regularizer: Union[str, Callable] = None,
        context_initializer: Union[str, Callable] = "glorot_uniform",
        context_regularizer: Union[str, Callable] = None,
        **kwargs,
    ):
        super(AttentionLayer, self).__init__(**kwargs)

        # ATTRIBUTES PARAMETERS
        self._projection_units = projection_units
        self._dropout = dropout

        self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
        self._kernel_constraint = tf.keras.constraints.get(kernel_constraint)

        self._bias_initializer = tf.keras.initializers.get(bias_initializer)
        self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
        self._bias_constraint = tf.keras.constraints.get(bias_constraint)

        self._activity_regularizer = tf.keras.regularizers.get(activity_regularizer)

        self._context_initializer = tf.keras.initializers.get(context_initializer)
        self._context_regularizer = tf.keras.regularizers.get(context_regularizer)

        # LAYERS
        self.W = tf.keras.layers.Dense(
            units=self._projection_units,
            activation="tanh",
            use_bias=True,
            kernel_initializer=self._kernel_initializer,
            bias_initializer=self._bias_initializer,
            kernel_regularizer=self._kernel_regularizer,
            bias_regularizer=self._bias_regularizer,
            activity_regularizer=self._activity_regularizer,
            kernel_constraint=self._kernel_constraint,
            bias_constraint=self._bias_constraint,
        )
        self.u = tf.keras.layers.Dense(
            units=1,
            use_bias=False,
            kernel_initializer=self._context_initializer,
            kernel_regularizer=self._context_regularizer,
        )
        self.dropout = tf.keras.layers.Dropout(self._dropout)

    def call(
        self,
        inputs: FloatTensorLike,
        training: bool = False,
        mask: TensorLike = None,
        return_attention_scores: bool = True,
    ) -> Tuple[FloatTensorLike]:

        attention_logits = self.u(self.W(inputs))
        attention_scores = tf.nn.softmax(attention_logits, axis=-2)

        # weighted_vectors = tf.multiply(attention_weights, inputs)
        weighted_vectors = attention_scores * inputs
        context_vector = tf.reduce_sum(weighted_vectors, axis=-2)

        if return_attention_scores:
            return context_vector, attention_scores

        return context_vector

    def get_config(self):
        config = super(AttentionLayer, self).get_config()
        config.update(
            {
                "projection_units": self._projection_units,
                "dropout": self._dropout,
                "kernel_initializer": tf.keras.initializers.serialize(
                    self._kernel_initializer
                ),
                "kernel_regularizer": tf.keras.regularizers.serialize(
                    self._kernel_regularizer
                ),
                "kernel_constraint": tf.keras.constraints.serialize(
                    self._kernel_constraint
                ),
                "bias_initializer": tf.keras.initializers.serialize(
                    self._bias_initializer
                ),
                "bias_regularizer": tf.keras.regularizers.serialize(
                    self._bias_regularizer
                ),
                "bias_constraint": tf.keras.constraints.serialize(
                    self._bias_constraint
                ),
                "activity_regularizer": tf.keras.regularizers.serialize(
                    self._activity_regularizer
                ),
                "context_initializer": tf.keras.initializers.serialize(
                    self._context_initializer
                ),
                "context_regularizer": tf.keras.regularizers.serialize(
                    self._context_regularizer
                ),
            }
        )
        return config


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

    Args:
        vocabulary_size (int): [description]
        n_classes (int): [description]
        attention_units (int, optional): [description]. Defaults to 100.
        word_recurrent_units (int, optional): [description]. Defaults to 100.
        sentence_recurrent_units (int, optional): [description]. Defaults to 100.
        embed_dimension (int, optional): [description]. Defaults to 200.
        initializer (Union[str, Callable], optional): [description].
            Defaults to "uniform".

    Call Args:
        x (TensorLike): [description]

    Returns:
        Dict[str, FloatTensorLike]: [description]
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
        initializer: Union[str, Callable] = "uniform",
        **kwargs,
    ):
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
