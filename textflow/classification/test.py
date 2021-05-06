"""Test for Attention layer."""

import tensorflow as tf
from .attention import AttentionLayer
from .han import HierarchicalAttentionNetwork


def test_attention():
    batch1 = tf.random.normal((16, 10, 50, 128))
    batch2 = tf.random.normal((16, 10, 128))

    attention = AttentionLayer(units=64)

    (att_batch1, weights_batch1), (att_batch2, weights_batch2) = (
        attention(batch1),
        attention(batch2),
    )
    assert att_batch1.shape == (16, 10, 128) & weights_batch1.shape == (16, 10, 50, 1)
    assert att_batch2.shape == (16, 128) & weights_batch2.shape(16, 10, 1)


def test_han():
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