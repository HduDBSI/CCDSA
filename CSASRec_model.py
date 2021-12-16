"""
Updated on Dec 20, 2020

model: Self-Attentive Sequential Recommendation

@author: Ziyao Geng(zggzy1996@163.com)
"""
import tensorflow as tf
from keras.constraints import non_neg
from tensorflow.keras.regularizers import l2
from tensorflow.keras.layers import Layer, Dense, LayerNormalization, Dropout, Embedding, Input
import keras.backend as K

# from CSASRec_modules import *
from Informer_modules import *


class SASRec(tf.keras.Model):
    def __init__(self, feature_columns, lat_lon_map, time_slot, blocks=1, num_heads=1, ffn_hidden_unit=128,
                 dropout=0., maxlen=40, norm_training=True, causality=False, embed_reg=1e-6):
        """
        SASRec model
        :param item_fea_col: A dict contains 'feat_name', 'feat_num' and 'embed_dim'.
        :param blocks: A scalar. The Number of blocks.
        :param num_heads: A scalar. Number of heads.
        :param ffn_hidden_unit: A scalar. Number of hidden unit in FFN
        :param dropout: A scalar. Number of dropout.
        :param maxlen: A scalar. Number of length of sequence
        :param norm_training: Boolean. If True, using layer normalization, default True
        :param causality: Boolean. If True, using causality, default True
        :param embed_reg: A scalar. The regularizer of embedding
        """
        super(SASRec, self).__init__()
        # sequence length
        self.maxlen = maxlen
        # feature columns
        self.user_fea_col, self.item_fea_col, self.c_fea_col = feature_columns
        # embed_dim
        self.embed_dim = self.item_fea_col['embed_dim']
        self.lat_lon = tf.cast(tf.convert_to_tensor(lat_lon_map), dtype=tf.float32)
        # d_model must be the same as embedding_dim, because of residual connection
        self.d_model = self.embed_dim
        # user embedding
        self.user_embedding = Embedding(input_dim=self.user_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.user_fea_col['embed_dim'],
                                        mask_zero=False,
                                        embeddings_initializer='random_uniform',
                                        embeddings_regularizer=l2(embed_reg))
        # item embedding
        self.item_embedding = Embedding(input_dim=self.item_fea_col['feat_num'],
                                        input_length=1,
                                        output_dim=self.item_fea_col['embed_dim'],
                                        mask_zero=True,
                                        embeddings_initializer='random_uniform',
                                        embeddings_regularizer=l2(embed_reg))
        self.pos_embedding = Embedding(input_dim=self.maxlen,
                                       input_length=1,
                                       output_dim=self.embed_dim,
                                       mask_zero=False,
                                       embeddings_initializer='random_uniform',
                                       embeddings_regularizer=l2(embed_reg))
        self.week_embedding = Embedding(input_dim=7,
                                        input_length=1,
                                        output_dim=self.embed_dim,
                                        mask_zero=False,
                                        embeddings_initializer='random_uniform',
                                        embeddings_regularizer=l2(embed_reg))
        self.hour_embedding = Embedding(input_dim=time_slot,
                                        input_length=1,
                                        output_dim=self.embed_dim,
                                        mask_zero=False,
                                        embeddings_initializer='random_uniform',
                                        embeddings_regularizer=l2(embed_reg))
        self.category_embedding = Embedding(input_dim=self.c_fea_col['feat_num'],
                                            input_length=1,
                                            output_dim=self.embed_dim,
                                            mask_zero=True,
                                            embeddings_initializer='random_uniform',
                                            embeddings_regularizer=l2(embed_reg))
        self.dense = tf.keras.layers.Dense(
            units=1,
            activation=None,
            kernel_initializer=tf.zeros_initializer(),
            bias_initializer=tf.zeros_initializer(),
            kernel_constraint=non_neg()
        )
        self.dropout = Dropout(dropout)
        # attention block
        self.encoder_layer = [EncoderLayer(self.d_model, num_heads, ffn_hidden_unit,
                                           dropout, norm_training, causality) for b in range(blocks)]
        self.c_encoder_layer = [EncoderLayer(self.d_model, num_heads, ffn_hidden_unit,
                                           dropout, norm_training, causality) for b in range(blocks)]
        # self.total_att = MultiHeadAttention(self.d_model*2, num_heads, causality=False)
        self.output_dense = Dense(1, activation=None)
        self.distance_dense = Dense(1, activation=None)
        # self.d_layerNormalization = LayerNormalization(epsilon=1e-6, trainable=norm_training)
        # self.t_layerNormalization = LayerNormalization(epsilon=1e-6, trainable=norm_training)

    def log2feats(self, seq_inputs, time_seq_inputs, w_seq_inputs, h_seq_inputs, c_seq_inputs, mask):
        # seq info
        seq_embed = self.item_embedding(seq_inputs)  # (None, maxlen, dim)
        seq_embed = tf.clip_by_norm(seq_embed, 1, -1)
        # context info
        w_seq_embed = self.week_embedding(w_seq_inputs)  # (None, maxlen, dim)
        w_seq_embed = tf.clip_by_norm(w_seq_embed, 1, -1)
        h_seq_embed = self.hour_embedding(h_seq_inputs)  # (None, maxlen, dim)
        h_seq_embed = tf.clip_by_norm(h_seq_embed, 1, -1)
        c_seq_embed = self.category_embedding(c_seq_inputs)  # (None, maxlen, dim)
        c_seq_embed = tf.clip_by_norm(c_seq_embed, 1, -1)
        # seq_embed = seq_embed + w_seq_embed + h_seq_embed + m_seq_embed
        seq_embed = seq_embed + h_seq_embed + w_seq_embed
        # pos_encoding = positional_encoding(seq_inputs, self.embed_dim)
        pos_encoding = tf.expand_dims(self.pos_embedding(tf.range(self.maxlen)), axis=0)
        seq_embed += pos_encoding
        seq_embed = self.dropout(seq_embed)
        # poi attention
        att_outputs = seq_embed  # (None, maxlen, dim)
        # category attention
        c_att_outputs = c_seq_embed
        att_outputs *= mask
        c_att_outputs *= mask

        # poi self-attention
        for block in self.encoder_layer:
            att_outputs = block([att_outputs, mask])  # (None, seq_len, dim)
            att_outputs *= mask

        # poi self-attention
        for block in self.c_encoder_layer:
            c_att_outputs = block([c_att_outputs, mask])  # (None, seq_len, dim)
            c_att_outputs *= mask

        # user_info = tf.reduce_mean(att_outputs, axis=1)  # (None, dim)
        user_info = tf.expand_dims(att_outputs[:, -1], axis=1)  # (None, 1, dim)
        c_user_info = tf.expand_dims(c_att_outputs[:, -1], axis=1)  # (None, 1, dim)

        return user_info, c_user_info

    def call(self, inputs, training=None, **kwargs):
        # # inputs
        user_input, seq_inputs, time_seq_inputs, time_input, w_seq_inputs, w_input, h_seq_inputs, h_input, c_seq_inputs, c_pos_input, c_neg_input, pos_inputs, neg_inputs = inputs  # (None, maxlen), (None, 1), (None, 1)
        # mask
        mask = tf.expand_dims(tf.cast(tf.not_equal(seq_inputs, 0), dtype=tf.float32), axis=-1)  # (None, maxlen, 1)
        user_info, c_user_info = self.log2feats(seq_inputs, time_seq_inputs, w_seq_inputs, h_seq_inputs, c_seq_inputs, mask)  # (None, 2, dim)
        pos_info = self.item_embedding(pos_inputs)  # (None, 1, dim)
        pos_info = tf.clip_by_norm(pos_info, 1, -1)
        neg_info = self.item_embedding(neg_inputs)  # (None, 1, dim)
        neg_info = tf.clip_by_norm(neg_info, 1, -1)

        # distance info
        seq_loc = tf.nn.embedding_lookup(self.lat_lon, seq_inputs)
        pos_loc = tf.nn.embedding_lookup(self.lat_lon, pos_inputs)
        neg_loc = tf.nn.embedding_lookup(self.lat_lon, neg_inputs)
        # # d info
        # pos_d = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(seq_loc - pos_loc), 2)),axis=-1)
        # pos_d = pos_d * mask
        # pos_d = tf.exp(-self.dense(self.d_layerNormalization(pos_d)))
        # pos_d = tf.reduce_mean(pos_d, axis=1)
        #
        # neg_d = tf.expand_dims(tf.sqrt(tf.reduce_sum(tf.square(seq_loc - neg_loc), 2)),axis=-1)
        # neg_d = neg_d * mask
        # neg_d = tf.exp(-self.dense(self.d_layerNormalization(neg_d)))
        # neg_d = tf.reduce_mean(neg_d, axis=1)  # [N,1,1]

        # context info
        w_embed = self.week_embedding(w_input)
        h_embed = self.hour_embedding(h_input)  # (None, maxlen, dim)
        c_pos_embed = self.category_embedding(c_pos_input) + w_embed + h_embed
        c_neg_embed = self.category_embedding(c_neg_input) + w_embed + h_embed
        #
        pos_info = pos_info + w_embed + h_embed
        neg_info = neg_info + w_embed + h_embed
        #
        # item_logits = tf.reduce_sum(user_info * pos_info, axis=-1)  # (None, 1)
        pos_logits = self.output_dense(tf.concat([user_info*pos_info,c_user_info*c_pos_embed], axis=-1))
        # pos_logits = tf.reduce_sum(pos_logits, axis=-1) + self.distance_dense(pos_d)
        pos_logits = tf.reduce_sum(pos_logits, axis=-1)
        neg_logits = self.output_dense(tf.concat([user_info * neg_info, c_user_info * c_neg_embed], axis=-1))
        # neg_logits = tf.reduce_sum(neg_logits, axis=-1) + self.distance_dense(neg_d)
        neg_logits = tf.reduce_sum(neg_logits, axis=-1)
        # loss
        losses = tf.reduce_mean(- tf.math.log(tf.nn.sigmoid(pos_logits)) -
                                tf.math.log(1 - tf.nn.sigmoid(neg_logits))) / 2
        self.add_loss(losses)
        # K.print_tensor(pos_logits)
        # K.print_tensor(neg_logits)
        # K.print_tensor(losses)
        # logits = tf.concat([pos_logits, neg_logits], axis=-1)
        return pos_logits,neg_logits

    def summary(self, **kwargs):
        user_inputs = Input(shape=(1,), dtype=tf.int32)
        seq_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        seq_t_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        t_inputs = Input(shape=(1,), dtype=tf.int32)
        seq_w_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        w_inputs = Input(shape=(1,), dtype=tf.int32)
        seq_h_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        h_inputs = Input(shape=(1,), dtype=tf.int32)
        seq_c_inputs = Input(shape=(self.maxlen,), dtype=tf.int32)
        c_pos_input = Input(shape=(1,), dtype=tf.int32)
        c_neg_input = Input(shape=(1,), dtype=tf.int32)
        pos_inputs = Input(shape=(1,), dtype=tf.int32)
        neg_inputs = Input(shape=(1,), dtype=tf.int32)
        tf.keras.Model(inputs=[user_inputs, seq_inputs,
                               seq_t_inputs, t_inputs,
                               seq_w_inputs, w_inputs,
                               seq_h_inputs, h_inputs,
                               seq_c_inputs, c_pos_input, c_neg_input,
                               pos_inputs, neg_inputs],
                       outputs=self.call([user_inputs, seq_inputs,
                                          seq_t_inputs, t_inputs,
                                          seq_w_inputs, w_inputs,
                                          seq_h_inputs, h_inputs,
                                          seq_c_inputs, c_pos_input, c_neg_input,
                                          pos_inputs, neg_inputs])
                       ).summary()


def test_model():
    item_fea_col = {'feat': 'item_id', 'feat_num': 100, 'embed_dim': 8}
    model = SASRec(item_fea_col, num_heads=8)
    model.summary()

# test_model()
