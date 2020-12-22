import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


# Source: https://keras.io/examples/nlp/text_classification_with_transformer/
class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}")
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # inputs.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)
        key = self.key_dense(inputs)
        value = self.value_dense(inputs)
        query = self.separate_heads(query, batch_size)
        key = self.separate_heads(key, batch_size)
        value = self.separate_heads(value, batch_size)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(attention, (batch_size, -1, self.embed_dim))
        output = self.combine_heads(concat_attention)
        return output


class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        # NEED tf-nightly for this package import tf.keras.layers MultiHeadAttention
        # self.att = MultiHeadAttention(num_heads, embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim)]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        # inputs_with_position = self._add_position_embedding(inputs)
        inputs_with_position = inputs
        attn_output = self.att(inputs_with_position)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs_with_position + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

#https://www.tensorflow.org/tutorials/text/transformer 
#    def positional_encoding(self, position, d_model):
#      angle_rads = get_angles(np.arange(position)[:, np.newaxis],
#                              np.arange(d_model)[np.newaxis, :],
#                              d_model)
#
#      # apply sin to even indices in the array; 2i
#      angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
#
#      # apply cos to odd indices in the array; 2i+1
#      angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
#
#      pos_encoding = angle_rads[np.newaxis, ...]
#
#      return tf.cast(pos_encoding, dtype=tf.float32)
##     def _add_position_embedding(self, inputs):
#         # inputs.shape = [batch_size, seq_len, embedding_dim]
#         batch_size, seq_len, embedding_dim = inputs.shape
# 
#         position_embedding = np.zeros_like(inputs)
#         for pos in range(seq_len):
#             for index in range(embedding_dim):
#                 i = index // 2
#                 divisor = np.power(10000, (2 * i / np.float32(embedding_dim)))
#                 if index % 2 == 0:
#                     for batch in range(batch_size):
#                         position_embedding[batch, pos, index] = np.sin(pos / divisor)
#                 else:
#                     for batch in range(batch_size):
#                         position_embedding[batch, pos, index] = np.cos(pos / divisor)
#         return inputs + position_embedding
# 
