"""
Created on Sept 10, 2020

modules of SASRec: attention mechanism, multi-head attention, self-attention block

@author: Ziyao Geng
"""
import numpy as np
import tensorflow as tf
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Embedding, Conv1D


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(seq_inputs, embed_dim):
    angle_rads = get_angles(np.arange(seq_inputs.shape[-1])[:, np.newaxis],
                            np.arange(embed_dim)[np.newaxis, :], embed_dim)
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


class CausalityMask():
    def __init__(self, B, H, L, index, scores):
        _mask_ex = 1-tf.linalg.band_part(tf.ones([B, H, L, tf.shape(scores)[-1]]),-1,0)
        indicator = tf.gather(_mask_ex, index, axis=-2, batch_dims=1)
        self._mask = tf.reshape(indicator, tf.shape(scores))
    @property
    def mask(self):
        return self._mask


def prob_QK(Q, K, num_head, factor):  # n_top: c*ln(L_q)
    B = tf.shape(Q)[0]
    H = tf.shape(Q)[1]
    L_Q = tf.shape(Q)[2]
    L_K = tf.shape(K)[2]
    H = num_head

    sample_k = tf.cast(factor * tf.math.ceil(tf.math.log(tf.cast(L_K, dtype=tf.float32))), dtype=tf.int32)  # c*ln(L_k)
    n_top = tf.cast(factor * tf.math.ceil(tf.math.log(tf.cast(L_Q, dtype=tf.float32))), dtype=tf.int32)  # c*ln(L_q)

    # K_expand = tf.tile(tf.expand_dims(K, -3), [1, 1, L_Q, 1, 1])
    index_sample = tf.random.uniform((L_Q, sample_k), maxval=L_K, dtype=tf.int32)
    K_sample = tf.gather(K,index_sample, axis=-2)
    Q_K_sample = tf.squeeze(tf.matmul(tf.expand_dims(Q, -2), tf.transpose(K_sample, [0, 1, 2, 4, 3])))
    # find the Top_k query with sparisty measurement
    M = tf.reduce_max(Q_K_sample, axis=-1) - (tf.reduce_sum(Q_K_sample, axis=-1) / tf.cast(L_K, dtype=tf.float32))
    M_top = tf.nn.top_k(M, n_top, sorted=False)[1]
    # use the reduced Q to calculate Q_K
    Q_reduce = tf.gather(Q,M_top, axis=-2, batch_dims=1)
    Q_K = tf.matmul(Q_reduce, tf.transpose(K, [0, 1, 3, 2]))  # factor*ln(L_q)*L_k
    return Q_K, M_top


def get_initial_context(V, L_Q, mask_flag):
    B = tf.shape(V)[0]
    H = tf.shape(V)[1]
    L_V = tf.shape(V)[2]
    E = tf.shape(V)[3]
    if not mask_flag:
        # V_sum = V.sum(dim=-2)
        V_sum = tf.reduce_mean(V, axis=-2)
        context = tf.tile(tf.expand_dims(V_sum, axis=-2),[1, 1, L_Q, 1])
    else:  # use mask
        # assert (L_Q == L_V)  # requires that L_Q == L_V, i.e. for self-attention only
        # cumsum 沿着某一个维度求累加和
        context = tf.cumsum(V, axis=-2)
    return context


def update_context(context_in, V, scores, index, L_Q, mask_flag):
    B = tf.shape(V)[0]
    H = tf.shape(V)[1]
    L_V = tf.shape(V)[2]
    E = tf.shape(V)[3]

    if mask_flag:
        attn_mask = CausalityMask(B, H, L_Q, index, scores)
        paddings = tf.ones_like(attn_mask.mask) * (-2 ** 32 + 1)
        scores = tf.where(tf.equal(attn_mask.mask, 1), paddings, scores)

    attn = tf.nn.softmax(logits=scores) # nn.Softmax(dim=-1)(scores)

    idx = tf.tile(tf.expand_dims(tf.range(B), -1), [1, tf.shape(index)[1]])
    n_idx = tf.tile(tf.expand_dims(tf.range(H), -1), [B, tf.shape(index)[1]])
    indices = tf.expand_dims(tf.stack([idx, n_idx, index], axis=-1), 1)
    output = tf.matmul(attn, V)
    index = tf.expand_dims(index, -1)
    context_in = tf.tensor_scatter_nd_update(context_in, indices, output)

    return context_in, None


def scaled_dot_product_attention(q, k, v, mask, causality=True, c=5):
    """
    Attention Mechanism
    :param q: A 3d tensor with shape of (None, seq_len, depth), depth = d_model // num_heads
    :param k: A 3d tensor with shape of (None, seq_len, depth)
    :param v: A 3d tensor with shape of (None, seq_len, depth)
    :param mask:
    :param causality: Boolean. If True, using causality, default True
    :return:
    """

    B = tf.shape(q)[0]
    H = tf.shape(q)[1]
    L_Q = tf.shape(q)[2]
    E = tf.shape(q)[3]
    L_K = tf.shape(k)[2]

    scores_top, index = prob_QK(q, k, H, c)
    scores_top = scores_top * (1. / tf.sqrt(tf.cast(E, dtype=tf.float32)))

    # get the context
    context = get_initial_context(v, L_Q, causality)
    # update the context with selected top_k queries
    context, attn = update_context(context, v, scores_top, index, L_Q, causality)

    return context


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads, causality=True, c=5):
        """
        Multi Head Attention Mechanism
        :param d_model: A scalar. The self-attention hidden size.
        :param num_heads: A scalar. Number of heads. If num_heads == 1, the layer is a single self-attention layer.
        :param causality: Boolean. If True, using causality, default True

        """
        super(MultiHeadAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.causality = causality
        self.c = c

        self.wq = Dense(d_model, activation=None)
        self.wk = Dense(d_model, activation=None)
        self.wv = Dense(d_model, activation=None)

    def call(self, q, k, v, mask):
        q = self.wq(q)  # (None, seq_len, d_model)
        k = self.wk(k)  # (None, seq_len, d_model)
        v = self.wv(v)  # (None, seq_len, d_model)
        B = tf.shape(q)[0]
        L_Q = tf.shape(q)[1]
        D = tf.shape(q)[2]
        L_K = tf.shape(k)[1]
        L_V = tf.shape(k)[1]
        q = tf.keras.backend.reshape(q, (-1, self.num_heads, L_Q, D//self.num_heads))
        k = tf.keras.backend.reshape(k, (-1, self.num_heads, L_K, D//self.num_heads))
        v = tf.keras.backend.reshape(v, (-1, self.num_heads, L_K, D//self.num_heads))

        # # split d_model into num_heads * depth, and concatenate
        # q = tf.reshape(tf.concat([tf.split(q, self.num_heads, axis=2)], axis=0),
        #                (-1, q.shape[1], q.shape[2] // self.num_heads))  # (None * num_heads, seq_len, d_model // num_heads)
        # k = tf.reshape(tf.concat([tf.split(k, self.num_heads, axis=2)], axis=0),
        #                (-1, k.shape[1], k.shape[2] // self.num_heads))  # (None * num_heads, seq_len, d_model // num_heads)
        # v = tf.reshape(tf.concat([tf.split(v, self.num_heads, axis=2)], axis=0),
        #                (-1, v.shape[1], v.shape[2] // self.num_heads))  # (None * num_heads, seq_len, d_model // num_heads)


        # attention
        outputs = scaled_dot_product_attention(q, k, v, mask, self.causality, self.c)  # (None * num_heads, seq_len, d_model // num_heads)

        # Reshape
        outputs = tf.reshape(outputs, (-1,L_V,D))
        # outputs = tf.concat(tf.split(scaled_attention, self.num_heads, axis=0), axis=2)  # (N, seq_len, d_model)

        return outputs


class FFN(Layer):
    def __init__(self, hidden_unit, d_model):
        """
        Feed Forward Network
        :param hidden_unit: A scalar. W1
        :param d_model: A scalar. W2
        """
        super(FFN, self).__init__()
        self.conv1 = Conv1D(filters=hidden_unit, kernel_size=1, activation='relu', use_bias=True)
        self.conv2 = Conv1D(filters=d_model, kernel_size=1, activation=None, use_bias=True)

    def call(self, inputs):
        x = self.conv1(inputs)
        output = self.conv2(x)
        return output


class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads=1, ffn_hidden_unit=128, dropout=0., norm_training=True, causality=True, c=10):
        """
        Encoder Layer
        :param d_model: A scalar. The self-attention hidden size.
        :param num_heads: A scalar. Number of heads.
        :param ffn_hidden_unit: A scalar. Number of hidden unit in FFN
        :param dropout: A scalar. Number of dropout.
        :param norm_training: Boolean. If True, using layer normalization, default True
        :param causality: Boolean. If True, using causality, default True
        """
        super(EncoderLayer, self).__init__()
        self.mha = MultiHeadAttention(d_model, num_heads, causality, c)
        self.ffn = FFN(ffn_hidden_unit, d_model)

        self.layernorm1 = LayerNormalization(epsilon=1e-6, trainable=norm_training)
        self.layernorm2 = LayerNormalization(epsilon=1e-6, trainable=norm_training)

        self.dropout1 = Dropout(dropout)
        self.dropout2 = Dropout(dropout)

    def call(self, inputs):
        x, mask = inputs
        # self-attention
        att_out = self.mha(x, x, x, mask)  # （None, seq_len, d_model)
        att_out = self.dropout1(att_out)
        # residual add
        out1 = self.layernorm1(x + att_out)
        # ffn
        ffn_out = self.ffn(out1)
        ffn_out = self.dropout2(ffn_out)
        # residual add
        out2 = self.layernorm2(out1 + ffn_out)  # (None, seq_len, d_model)
        return out2