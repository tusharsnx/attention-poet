import tensorflow as tf
from tensorflow.keras.callbacks import Callback


## callback to generate seq at end of each epoch
class SeqGenerateCallback(Callback):
    def __init__(self, trigger_seq):
        super().__init__()
        self.trigger_seq = trigger_seq
        
    def on_epoch_end(self, epoch, logs=None):
        curr_seq = self.trigger_seq.numpy()
        for i in range(self.model.preprocessor.seq_len):
            next_id = tf.argmax(self.model(curr_seq), axis=-1)[:,i]
            curr_seq[:, i] = next_id
        print(f"after epoch {epoch} model generates:")
        print("actual sequence: ", curr_seq)
        print("generated text sequence: ", self.model.preprocessor.get_text(curr_seq))
        