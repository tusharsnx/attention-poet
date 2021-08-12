# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'

import tensorflow as tf
from typing import Dict, List
import numpy as np
import pickle as pkl


EMBEDDING_DIMS=100
MAX_VOCAB_SIZE = 10000
MAX_NEG_SAMPLES = 3
MAX_SEQ_LEN = MAX_NEG_SAMPLES+2     # +2 for target and context words within the sequence


class Preprocessor:
    def __init__(self, max_vocab_size):
        self.max_vocab_size = max_vocab_size
        # self.seq_len = seq_len

        # fills after adapt
        self.vocab_size = None
        self.vocab = None
        self.word_ids: Dict[str, int] = None
        self.rev_word_ids:  Dict[int, str] = None
        
        self.string_lookup = tf.keras.layers.experimental.preprocessing.StringLookup(max_tokens=self.max_vocab_size)
    
    # bypass or create custom standardize
    def _custom_standardize(self, text):
        ''' Implements custom standardizing strategy
        '''
        return text

    # builds lookup layer's vocabulary(calls adapt())
    def build_vocab(self, inputs) -> List[str]:

        inputs = tf.constant(inputs, dtype=tf.string)
        assert tf.rank(inputs)==2, "inputs rank must be 2, add or reduce extra axis"

        self.string_lookup.adapt(inputs)
        self.vocab = self.string_lookup.get_vocabulary()
        self.vocab_size = self.string_lookup.vocabulary_size()
        self._build_dictionary(self.vocab)

    # utility to build word_ids
    def _build_dictionary(self, vocab_list: List[str]) -> None:
        word_ids = dict()
        rev_word_ids = dict()
        for i, item in enumerate(vocab_list):
            word_ids[item] = i
            rev_word_ids[i] = item
        self.word_ids = word_ids
        self.rev_word_ids = rev_word_ids

    # 
    def __call__(self, inputs):
        inputs = tf.constant(inputs, dtype=tf.string)
        assert tf.rank(inputs)==2, "inputs rank must be 2, add or reduce extra axis"
        int_tokens = self.string_lookup(inputs)                   # (None, num_tokens)
        return int_tokens                            

    
    # get text back from seq like [[5,2,3,4,0,0,0]]
    def get_text(self, seqs) -> List[List[str]]:
        texts = []
        for seq in seqs:
            string = ""
            for ids in seq:
                if ids!=0:
                    string+= " " + self.rev_word_ids[ids]
                else:
                    break
            texts.append([string.strip(" ")])
        return tf.Tensor(texts, dtype=tf.string)



with open("embedding_data.pkl", "rb") as f:
    inputs = pickle.load(f)
    

# inputs = inputs.reshape((-1, 1))
x_train, y_train = np.array(inputs[0]), np.array(inputs[1])
x_train.shape, y_train.shape



preprocessor = Preprocessor(max_vocab_size=MAX_VOCAB_SIZE)
preprocessor.build_vocab(x_train)
preprocessed_inputs = preprocessor(x_train)
print(preprocessor.vocab[:3])



preprocessed_inputs, y_train


class Embedder(tf.keras.models.Model):
    def __init__(self, preprocessor, embedding_dims=100):
        super().__init__()
        self.embedding_dims = embedding_dims
        self.preprocessor = preprocessor
        self.vocab_size = self.preprocessor.vocab_size
        self.emb_word = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dims)
        self.emb_contxt = tf.keras.layers.Embedding(input_dim=self.vocab_size, output_dim=self.embedding_dims)
        
    # def build(self, input_shape):
    #     assert input_shape.rank==2
    #     self.vocab_size = input_shape[1]
    #     self.output_layer = tf.keras.layers.Dense(self.vocab_size, activation="softmax")
    #     print(self.vocab_size)

    def call(self, inputs):
        # print(inputs.numpy())
        words = inputs[:, :-1]
        contxts = inputs[:, -1:]
        # print(words, contxts)
        words_emb = self.emb_word(words)
        contxts_emb = self.emb_contxt(contxts)
        # print(words_emb, contxts_emb)
        product = tf.multiply(words_emb, contxts_emb)
        # print(product)
        dots = tf.reduce_sum(product, axis=2)
        # print(dots)
        return dots



model_checkpoints = tf.keras.callbacks.ModelCheckpoint(
    monitor="loss",
    filepath="chkpt/",
    save_best_only=True, 
    save_weights_only=True
)


model = Embedder(embedding_dims=EMBEDDING_DIMS, preprocessor=preprocessor)


model.compile(loss=tf.keras.losses.BinaryCrossentropy(), optimizer=tf.keras.optimizers.Adam(), metrics=["accuracy"], run_eagerly=True)
model.fit(preprocessed_inputs, y_train, epochs=10, batch_size=32, callbacks=[model_checkpoints])


model.load_weights("Untitled Folder/chkpt")


embeddings = model.emb_word.weights[0].numpy()
emb_dict = dict()
for word, idx in preprocessor.word_ids.items():
    if idx==0:
        emb_dict["pad"] = list(embeddings[idx])
    else:
        emb_dict[word] = list(embeddings[idx])


with open(f"embedding-{EMBEDDING_DIMS}.pkl", "wb") as f:
    pkl.dump(emb_dict, f)


