import numpy as np
import tensorflow as tf
from tensorflow.random import categorical
from typing import Dict, List


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

        # parameters
        self.d_model = d_model  # model dims 
        self.dff = dff  # ffn dense layer units
        self.heads = heads  # number of heads

        # layers
        self.ffn = FFN(d_model, dff)
        self.ln1 = tf.keras.layers.LayerNormalization()
        self.ln2 = tf.keras.layers.LayerNormalization()
        self.wq = tf.keras.layers.Dense(self.d_model)
        self.wv = tf.keras.layers.Dense(self.d_model)
        self.dropout1 = tf.keras.layers.Dropout(rate)
        self.dropout2 = tf.keras.layers.Dropout(rate)
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads=self.heads, key_dim=self.d_model)


    def call(self, inputs, training=False, mask=None):
        q = self.wq(inputs)                                                                 # (None, seq_len, d_model)
        v = self.wv(inputs)                                                                 # (None, seq_len, d_model)

        # mask is 1 for keeping and 0 for removing
        attention_outputs = self.mha(query=q, value=v, attention_mask=mask)                 # (None, query_len, d_model)
        dropped_attention_outputs = self.dropout1(attention_outputs, training=training)
        outputs1 = self.ln1(inputs+dropped_attention_outputs)

        ffn_outputs = self.ffn(outputs1)                                                    # (None, query_len, d_model)
        dropped_ffn_outputs = self.dropout1(ffn_outputs, training=training)
        outputs = self.ln2(outputs1+dropped_ffn_outputs)                                    # (None, query_len, d_model)
        
        return outputs



class Poet(tf.keras.models.Model):
    def __init__(self, preprocessor, num_blocks=1, d_model=256, 
        dff=512, heads=8,rate=0.1):
        super().__init__()

        # parameters
        self.d_model = d_model
        self.preprocessor = preprocessor
        self.num_blocks = num_blocks

        # generating pos encoding now to save time while calling call()(as it is constant for all examples)
        self.pos_encoding = self.positional_encoding()
        
        # layers
        self.embedding_layer = tf.keras.layers.Embedding(self.preprocessor.vocab_size, self.d_model)
        self.dropout = tf.keras.layers.Dropout(rate)
        self.blocks = [Block(d_model=self.d_model, dff=dff, heads=heads, rate=rate) for i in range(self.num_blocks)]
        self.final_layer = tf.keras.layers.Dense(self.preprocessor.vocab_size)


    def call(self, inputs, training=False):
        
        embeddings = self.embedding_layer(inputs)

        embeddings *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))

        # adding positional encoding
        x = embeddings + self.pos_encoding
        
        # generate lookahead mask
        mask = self.lookahead_mask(self.preprocessor.seq_len)

        x = self.dropout(x, training=training)

        # passing rich attention embedding through each block
        for block in self.blocks:
            x = block(x, training=training, mask=mask)
        
        outputs = self.final_layer(x)
        
        return outputs

    def generate(self, inputs, return_seq=False, value=1, sampling="top_k"):
        
        sampling_strategies = set({"top_k", "temperature", "top_p"})
        assert sampling in sampling_strategies, f"sampling value should be one of {sampling_strategies}"
        
        curr_seq = inputs.numpy()
        padded_pos = tf.math.equal(curr_seq, 0)

        if return_seq:
            probabs = []

        for i in range(self.preprocessor.seq_len):
            logits = self.call(curr_seq)[0, i:i+1, :]
            # shutting probabilities according to temperature
            # mask = tf.cast(tf.logical_not(tf.math.less(probab, 1-temperature)), dtype=tf.float32)
            # probab *= mask
            if sampling=="temperature":
                logits = self.temperature_sampling(logits, temperature=value)
            
            if sampling=="top_k":
                logits = self.top_k_sampling(logits, k=value)
            
            if sampling=="top_p":
                logits = self.top_p_sampling(logits, p=value)
            
            probab = tf.keras.activations.softmax(logits)
            
            if return_seq:
                probabs.append(probab)
            
            next_id = categorical(probab, 1)[0, 0]
            if padded_pos[:, i].numpy(): 
                curr_seq[:, i] = next_id

            # else:
            #     # shutting probabilities according to temperature
            #     probab = tf.keras.activations.softmax(logits)
            #     if return_seq:
            #         probabs.append(probab)
            #     mask = tf.cast(tf.logical_not(tf.math.less(probab, 1-value)), dtype=tf.float32)
            #     # print("current timestep probablities:\n", tf.reduce_sum(tf.logical_not(tf.math.equal(probab, 0))).numpy())
            #     # probab *= mask
            #     # print("current timestep probablities after mask:\n", tf.reduce_sum(tf.logical_not(tf.math.equal(probab, 0))).numpy())
            #     # next_id = categorical(probab, 1)[0, 0]
            #     next_id = tf.argmax(probab, axis=-1)
            #     if padded_pos[:, i].numpy(): 
            #         curr_seq[:, i] = next_id

        if return_seq:
            return self.preprocessor.get_text(curr_seq)[0, 0], curr_seq, probabs
        return self.preprocessor.get_text(curr_seq)[0, 0]

    
    def temperature_sampling(self, logits, temperature=1):
        assert temperature>0 and temperature<=1, "temperature should be between 0 and 1"
        return logits/temperature

    
    def top_k_sampling(self, logits, k=None):
        if k is None:
            k = logits.shape[-1]
        values, _ = tf.math.top_k(logits, k=k)
        not_top_k_indices = logits < tf.expand_dims(values[:, -1], -1)
        top_k_logits = self.set_value_on_indices(logits=logits, indices=not_top_k_indices, value=1e-9)
        return top_k_logits


    def top_p_sampling(self, logits, p=1):
        sorted_indices = tf.argsort(logits, direction="DESCENDING")
        # Flatten logits as tf.gather on TPU needs axis to be compile time constant.
        logits_shape = logits.shape
        range_for_gather = tf.expand_dims(tf.range(0, logits_shape[0]), axis=1)
        range_for_gather = tf.tile(range_for_gather * logits_shape[1],
                                    [1, logits_shape[1]]) + sorted_indices
        flattened_logits = tf.reshape(logits, [-1])
        flattened_sorted_indices = tf.reshape(range_for_gather, [-1])
        sorted_logits = tf.reshape(
            tf.gather(flattened_logits, flattened_sorted_indices),
            [logits_shape[0], logits_shape[1]])
        cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)

        # Remove tokens with cumulative probability above the threshold.
        sorted_indices_to_remove = cumulative_probs > p

        # Shift the indices to the right to keep the first token above threshold.
        sorted_indices_to_remove = tf.roll(sorted_indices_to_remove, 1, axis=-1)
        sorted_indices_to_remove = tf.concat([
            tf.zeros_like(sorted_indices_to_remove[:, :1]),
            sorted_indices_to_remove[:, 1:]
        ], -1)

        # Scatter sorted indices to original indexes.
        indices_to_remove = self.scatter_values_on_batch_indices(sorted_indices_to_remove, sorted_indices)
        
        top_p_logits = self.set_value_on_indices(logits, indices_to_remove, np.NINF)
        
        return top_p_logits

    
    @staticmethod
    def scatter_values_on_batch_indices(values, batch_indices):
        """Scatter `values` into a tensor using `batch_indices`.
        Args:
            values: tensor of shape [batch_size, vocab_size] containing the values to
                scatter
            batch_indices: tensor of shape [batch_size, vocab_size] containing the
                indices to insert (should be a permutation in range(0, n))
        Returns:
            Tensor of shape [batch_size, vocab_size] with values inserted at
            batch_indices
        """
        tensor_shape = batch_indices.shape
        broad_casted_batch_dims = tf.reshape(
            tf.broadcast_to(
                tf.expand_dims(tf.range(tensor_shape[0]), axis=-1), tensor_shape),
            [1, -1])
        pair_indices = tf.transpose(
            tf.concat([broad_casted_batch_dims,
                        tf.reshape(batch_indices, [1, -1])], 0))
        return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), tensor_shape)
    

    @staticmethod
    def set_value_on_indices(logits, indices, value):
        value = tf.zeros_like(logits) + value           # (seq_len, vocab_size)
        new_logits = tf.where(indices, value, logits)   # (seq_len, vocab_size)
        return new_logits

        

    @staticmethod
    def get_angles(pos, i, dims):
        angle_rates = 1 / (10000 ** ((2 * (i//2)) / dims))
        return pos * angle_rates
    
    
    # @staticmethod
    # def embedding_from_file(embeddings: Dict, word_ids: Dict, vocab_size: int, embedding_dims: int):
    #     embed = np.random.rand(vocab_size, embedding_dims)      # (vocab-size, embedding_dims)
    #     words = word_ids.keys()     # words in preprocessor's vocab list
    #     hits, misses = 0,0
    #     for word, emb in embeddings.items():
    #         if word in words:
    #             hits += 1
    #             embed[word_ids[word]] = emb
    #         else:
    #             misses+=1
    #     print(f"Embeddings hits: {hits}, misses: {misses} from the trained embeddings")
    #     return embed


    def positional_encoding(self):
        angle_rads = self.get_angles(np.arange(self.preprocessor.seq_len)[:, np.newaxis],
            np.arange(self.d_model)[np.newaxis, :],
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
        return tf.linalg.band_part(tf.ones((seq, seq)), -1, 0)

