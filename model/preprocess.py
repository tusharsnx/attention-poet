import tensorflow as tf
from typing import List, Dict
import tensorflow_text as tf_txt
import numpy as np

class Preprocessor:
    def __init__(self, vocab_size, seq_len=10):
        self.seq_len = seq_len
        self.vocab: List[str] = None
        self.word_ids: Dict[str, int] = None
        self.rev_word_ids:  Dict[int, str] = None
        self.vocab_size = None
        self.tokenizer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=vocab_size,
                                        output_sequence_length=self.seq_len, standardize=self.custom_standardize
                                        )

    def __call__(self, inputs):

        inputs = tf.constant(inputs)
        assert tf.rank(inputs)==2, "inputs rank must be 2, add or reduce extra axis"

        # encoding into utf8
        encoded_seq = tf_txt.normalize_utf8(inputs, "NFKD")

        # tokenizing into 'seq_len' num of tokens
        tokenized_seq = self.tokenizer(self.add_extra(encoded_seq))

        # adding end token back if gets cliped
        end_token = np.array([self.word_ids["[KHATAM]"]])[:, np.newaxis]
        tokenized_seq = tokenized_seq.numpy()
        tokenized_seq[(tokenized_seq[:, -1]==0)==0, -1] = end_token

        # returning as tensor
        return tf.constant(tokenized_seq)

    @staticmethod
    def add_extra(inputs):
        inputs = tf.constant(inputs)
        return [["[SURU] "]]+inputs+[[" [KHATAM]"]]
    
    def custom_standardize(self, text):
        ''' Implements custom standardizing strategy
        '''
        return text

    
    def build_vocab(self, inputs) -> List[str]:

        inputs = tf.constant(inputs)
        assert tf.rank(inputs)==2, "inputs rank must be 2, add or reduce extra axis"

        self.tokenizer.adapt(self.add_extra(inputs))
        self.vocab = self.tokenizer.get_vocabulary()
        self.vocab_size = len(self.vocab)
        self.build_dictionary(self.vocab)
        return self.vocab

    def build_dictionary(self, vocab_list: List[str]) -> None:
        word_ids = dict()
        rev_word_ids = dict()
        for i, item in enumerate(vocab_list):
            word_ids[item] = i
            rev_word_ids[i] = item
        self.word_ids = word_ids
        self.rev_word_ids = rev_word_ids

    def get_text(self, seqs) -> List[List[str]]:
        texts = []
        for seq in seqs:
            string = ""
            for ids in seq:
                if ids!=0 and ids!=self.word_ids["[KHATAM]"]:
                    if ids!=self.word_ids["[SURU]"]:
                        string+= " " + self.rev_word_ids[ids]
                else:
                    break
            texts.append([string.strip(" ")])
        return texts

