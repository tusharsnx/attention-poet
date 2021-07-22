import numpy as np
import tensorflow as tf
import pandas as pd
import tensorflow_text as tf_txt 
from typing import List, Dict
from model.preprocess import Preprocessor

MAX_VOCAB_SIZE = 10000
EMBEDDING_DIM = 200
DFF = 512
D_MODEL = 256
MAX_SEQ_LEN = 10

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
        self.d_model = d_model      # model dims 
        self.dff = dff                           # ffn dense layer units
        self.heads = heads               # number of heads

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

    # def build(self, input_shape):
    #     self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.heads, key_dim=d_model)

    def call(self, inputs, training=False, mask=None):
        q = self.wq(inputs)     #(None, seq_len, d_model)
        v = self.wv(inputs)      #(None, seq_len, d_model)

        # projecting on higher dimension to add with attention_outputs in  ln
        inputs = self.wi(inputs)  #(None, seq_len, d_model)
        attention_outputs = self.mha(query=q, value=v, attention_mask=mask)      # output shape (None, query_len, d_model)
        dropped_attention_outputs = self.dropout1(attention_outputs, training=training)
        outputs = self.ln1(inputs+dropped_attention_outputs)

        ffn_outputs = self.ffn(outputs)     # output shape (None, query_len, d_model)
        dropped_ffn_outputs = self.dropout1(ffn_outputs, training=training)
        outputs = self.ln2(inputs+dropped_ffn_outputs)        # output shape (None, query_len, d_model)
        
        return outputs


layer = Block(d_model=8, dff=256, heads=4)
mask = tf.keras.Input(shape=[4, 4])
source = tf.keras.Input(shape=[4, 100])
outputs = layer(inputs=source, mask=mask)
print(outputs.shape)


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

        pos_encoding = angle_rads[np.newaxis, ...]

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

## test code
preprocessor = Preprocessor(vocab_size=100, seq_len=10)

string_inputs = [["जैसा"], ["i am fine, what about you. ? i mean"]]

vocab = preprocessor.build_vocab(string_inputs)
print(vocab)

poet = Poet(preprocessor=preprocessor)

output = poet.lookahead_mask(4)
print("lookahead mask shape: ", output.shape)

preprocess_inputs = preprocessor(string_inputs)
print("preprocess inputs shape: ", preprocess_inputs.shape)

outputs = poet.call(preprocess_inputs)
print(outputs.shape)                                                    # shape (None, seq_len, vocab_size)
print("model outputs shape: ", outputs.shape)





## callback to generate seq at end of each epoch
class SeqGenerateCallback(tf.keras.callbacks.Callback):
    def __init__(self, trigger_seq):
        super().__init__()
        self.trigger_seq = trigger_seq
        
    def on_epoch_end(self, epoch, logs=None):
        curr_seq = self.trigger_seq.numpy()
        for i in range(self.model.preprocessor.seq_len):
            next_id = tf.argmax(self.model(curr_seq), axis=-1)[:,i]
            curr_seq[:, i] = next_id
        print(f"after epoch {epoch} models generates:")
        print("actual seq: ", curr_seq)
        print(self.model.preprocessor.get_text(curr_seq))
        


model = Poet(preprocessor=preprocessor, num_blocks=1, d_model=256, dff=512, heads=8, embedding_dims=100)
model.compile(loss=MaskedLoss(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=[MaskedAccuracy()])


string_inputs = [["i will love you"], ["i like you"]]
trigger_inputs = [["i"]]
vocab = preprocessor.build_vocab(string_inputs)
print(vocab)
preprocess_inputs = preprocessor(string_inputs)
trigger_inputs = preprocessor(trigger_inputs)
history = model.fit(x=preprocess_inputs, y=preprocess_inputs, batch_size=1, epochs=4, callbacks=[SeqGenerateCallback(trigger_inputs)])
