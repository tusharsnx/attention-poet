import numpy as np
import tensorflow as tf
from tensorflow.random import categorical


class FFN(tf.keras.layers.Layer):
    def  __init__(self, d_model, dff):
            super().__init__()
            self.dff = dff
            self.dense1 = tf.keras.layers.Dense(dff, activation="relu")
            self.dense2 = tf.keras.layers.Dense(d_model)

    def call(self, inputs):
            outputs = self.dense1(inputs)
            outputs = self.dense2(outputs)
            return outputs


class Block(tf.keras.layers.Layer):
    def __init__(self, d_model: int, dff: int = 2048, heads: int = 8, rate: int = 0.1):
        super().__init__()

        assert d_model%heads==0

        #parameters
        self.d_model = d_model          # model dims 
        self.dff = dff                            # ffn dense layer units
        self.heads = heads                  # number of heads

        #layers
        self.ffn = FFN(d_model, dff)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.wq = tf.keras.layers.Dense(self.d_model)
        self.wv = tf.keras.layers.Dense(self.d_model)
        self.wi = tf.keras.layers.Dense(self.d_model)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.heads, key_dim=self.d_model)


    def call(self, inputs, training=False, mask=None):
        q = self.wq(inputs)                                                                                                     #(None, seq_len, d_model)
        v = self.wv(inputs)                                                                                                      #(None, seq_len, d_model)

        # projecting on higher dimension to add with attention_outputs in  ln
        inputs = self.wi(inputs)                                                                                             # (None, seq_len, d_model)
        attention_outputs = self.mha(query=q, value=v, attention_mask=mask)                 # (None, query_len, d_model)
        dropped_attention_outputs = self.dropout1(attention_outputs, training=training)
        outputs = self.ln1(inputs+dropped_attention_outputs)

        ffn_outputs = self.ffn(outputs)                                                                                 # (None, query_len, d_model)
        dropped_ffn_outputs = self.dropout1(ffn_outputs, training=training)
        outputs = self.ln2(inputs+dropped_ffn_outputs)                                                      # (None, query_len, d_model)
        
        return outputs



class Poet(tf.keras.models.Model):
    def __init__(self, preprocessor, num_blocks=1, d_model=256, dff=512, heads=8, embedding_dims=100):
        super().__init__()
        self.d_model = d_model
        self.preprocessor = preprocessor
        self.num_blocks = num_blocks
        self.embedding_dims = embedding_dims

        # generating pos encoding now to save time while calling call()(as it is constant for all examples)
        self.pos_encoding = self.positional_encoding()

        self.embedding_layer = tf.keras.layers.Embedding(input_dim=self.preprocessor.vocab_size, 
                                            output_dim=self.embedding_dims, mask_zero=True, input_length=self.preprocessor.seq_len
                                            )
        self.blocks = [Block(d_model=self.d_model, dff=dff, heads=heads) for i in range(self.num_blocks)]

        self.final_layer = tf.keras.layers.Dense(self.preprocessor.vocab_size, activation="softmax")
    
    @staticmethod
    def get_angles(pos, i, dims):
        angle_rates = 1 / (10000 ** ((2 * (i//2)) / dims))
        return pos * angle_rates

    def positional_encoding(self):
        angle_rads = self.get_angles(np.arange(self.preprocessor.seq_len)[:, np.newaxis],
                                np.arange(self.embedding_dims)[np.newaxis, :],
                                self.d_model
                                )
                                
        # apply sin to even indices in the array; 2i
        angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

        # apply cos to odd indices in the array; 2i+1
        angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

        pos_encoding = angle_rads[np.newaxis, :]

        return tf.cast(pos_encoding, dtype=tf.float32)
    
    @staticmethod
    def lookahead_mask(seq):
        return 1 - tf.linalg.band_part(tf.ones((seq, seq)), -1, 0)


    def call(self, inputs):
        
        embeddings = self.embedding_layer(inputs)

        # adding positional encoding
        x = embeddings + self.pos_encoding
        
        # generate lookahead mask
        mask = self.lookahead_mask(self.preprocessor.seq_len)

        # passing rich attention embedding through each block
        for block in self.blocks:
            x = block(x, mask=mask)
        
        outputs = self.final_layer(x)
        
        return outputs

    def generate(self, inputs, return_seq=False):
        curr_seq = inputs.numpy()
        padded_pos = tf.math.equal(curr_seq, 0)

        for i in range(self.preprocessor.seq_len):
            next_id = categorical(self.call(curr_seq)[0, i:i+1, :], 1)[0, 0]
            if padded_pos[:, i].numpy(): 
                curr_seq[:, i] = next_id
        if return_seq:
            return self.preprocessor.get_text(curr_seq)[0, 0], curr_seq
        return self.preprocessor.get_text(curr_seq)[0, 0]
            # self.model.preprocessor.get_text(seq)[(0, 0)]

