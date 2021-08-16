import os
from typing import List, Dict

import tensorflow as tf
import tensorflow_text as tf_txt
import numpy as np
import tensorflow_text as tf_txt
from tensorflow_text.tools.wordpiece_vocab import bert_vocab_from_dataset as bert_vocab_builder


class Preprocessor:
    def __init__(self, max_vocab_size, seq_len=10, new=True):
        self.seq_len = seq_len
        self.max_vocab_size = max_vocab_size
        self.vocab: List[str] = None
        self.word_ids: Dict[str, int] = None
        self.rev_word_ids:  Dict[int, str] = None
        self.vocab_size = None
        self.use_new_tokenizer = new
        
        if not self.use_new_tokenizer:
            self.tokenizer = tf.keras.layers.experimental.preprocessing.TextVectorization(max_tokens=max_vocab_size,
                output_sequence_length=self.seq_len, standardize=self._custom_standardize
            )
        
        else:
            self.tokenizer = None
            self.reserved_tokens = ["[PAD]", "[UNK]", "[SURU]", "[KHATAM]"]


    def __call__(self, inputs, training=False):
        inputs = tf.constant(inputs, dtype=tf.string)
        assert tf.rank(inputs)==2, "inputs rank must be 2, add or reduce extra axis"
        
        if not self.use_new_tokenizer:
            # tokenizing into 'seq_len' num of tokens
            tokenized_seq = self.tokenizer(self._add_extra(inputs, training=training))  # (None, seq_len)
        
        else:
            inputs = self._add_extra(inputs, training=training)
            inputs = tf_txt.WhitespaceTokenizer().tokenize(inputs) # (None, 1, seq_len)
            inputs = tf.squeeze(inputs, axis=1) # (None, seq_len)
            ragged_tokenized_seq = self.tokenizer.tokenize(inputs)  # (None, seq_len, sub_words)
            ragged_tokenized_seq = ragged_tokenized_seq.merge_dims(-2, -1)  # (None, seq_len+subwords)
            tokenized_seq = self.pad_or_trim(ragged_tokenized_seq)  # (None, seq_len)

        if not training:
            return tokenized_seq                                                

        # adding end token back if gets cliped and adding padding_token to increase seq_len by 1
        tokenized_seq = tokenized_seq.numpy()

        end_token = self.word_ids["[KHATAM]"]
        extra_token_function = lambda x: end_token if (x[-1]!=0) else 0

        # generating token to be added at the last for the batch
        extra_token = np.apply_along_axis(extra_token_function, 1, tokenized_seq)

        # new tokenized_seq with increased sequence length(seq_len)
        tokenized_seq = np.hstack((tokenized_seq, extra_token[:, np.newaxis]))  # (None, seq_len+1)

        # returning as input tensor and target tensor shifted by 1 position
        return tf.constant(tokenized_seq[:, :-1]), tf.constant(tokenized_seq[:, 1:])    # (None, seq_len), (None, seq_len)

       
    def pad_or_trim(self, inputs):
        return inputs.to_tensor(default_value=0, shape=[None, self.seq_len])


    @staticmethod
    def _add_extra(inputs, training=False):
        start_token = tf.constant([["[SURU] "]], dtype=inputs.dtype)
        end_token = tf.constant([[" [KHATAM]"]], dtype=inputs.dtype)
        if training:
            data = start_token+inputs+end_token
            return data
        else:
            return start_token+inputs
    
    def _custom_standardize(self, text):
        ''' Implements custom standardizing strategy
        '''
        return text

    
    def build_vocab(self, inputs) -> List[str]:
        inputs = tf.constant(inputs, dtype=tf.string)
        assert tf.rank(inputs)==2, "inputs rank must be 2, add or reduce extra axis"
        
        if not self.use_new_tokenizer:
            self.tokenizer.adapt(self._add_extra(inputs, training=True))
            self.vocab = self.tokenizer.get_vocabulary()
            self.vocab_size = self.tokenizer.vocabulary_size()
            self._build_dictionary(self.vocab)
            return self.vocab
        
        else:
            inputs = tf.data.Dataset.from_tensor_slices(tf.reshape(inputs, shape=(-1)))
            self.vocab = bert_vocab_builder.bert_vocab_from_dataset(
                inputs,
                vocab_size=self.max_vocab_size,
                reserved_tokens=self.reserved_tokens,
            )
            self.vocab_size = len(self.vocab)
            self._build_dictionary(self.vocab)
            full_path = self.write_vocab_file(self.vocab)
            self.tokenizer = tf_txt.WordpieceTokenizer(full_path)
            return self.vocab


    def _build_dictionary(self, vocab_list: List[str]) -> None:
        word_ids = dict()
        rev_word_ids = dict()
        for i, item in enumerate(vocab_list):
            word_ids[item] = i
            rev_word_ids[i] = item
        self.word_ids = word_ids
        self.rev_word_ids = rev_word_ids

    def get_text(self, seqs, return_subtokens=False) -> List[List[str]]:
        texts = []
        
        if not self.use_new_tokenizer:
            for seq in seqs:
                string = ""
                for ids in seq:
                    if ids!=0 and ids!=self.word_ids["[KHATAM]"]:
                        if ids!=self.word_ids["[SURU]"]:
                            string+= " " + self.rev_word_ids[ids]
                    else:
                        break
                texts.append([string.strip(" ")])
            return np.array(texts)
        
        else:
            if return_subtokens:
                subtoken_texts = []
                subtoken_seqs = tf.gather(self.vocab, seqs).numpy()
                for seq in subtoken_seqs:
                    string = ""
                    for token in seq:
                        try:
                            token = str(token, encoding="utf-8")
                        except:
                            print(token)
                            raise Exception()
                        
                        if not token=="[KHATAM]":
                            if not token=="[SURU]":
                                string+= " " + token
                        
                        else:
                            break
                    
                    subtoken_texts.append([string.strip(" ")])
                
            seqs = self.tokenizer.detokenize(seqs).numpy()
            for seq in seqs:
                string = ""
                for token in seq:
                    token = str(token, encoding="utf-8")
                    
                    if not token=="[KHATAM]":
                        if not token=="[SURU]":
                            string+= " " + token
                    
                    else:
                        break
                
                texts.append([string.strip(" ")])

            if return_subtokens:
                return np.array(texts), np.array(subtoken_texts)
            
            else:
                return np.array(texts)

    @staticmethod
    def write_vocab_file(vocab, filepath=""):
        full_path = os.path.join(filepath,"vocab_file.txt")
        with open(full_path, 'w') as f:
            for token in vocab:
                print(token, file=f)
        return full_path
