from tensorflow.keras.callbacks import Callback

## callback to generate seq at end of each epoch
class SeqGenerateCallback(Callback):
    def __init__(self, trigger_seq):
        super().__init__()
        self.trigger_seq = trigger_seq
        
    def on_epoch_end(self, epoch, logs=None):
        seq = self.model.generate(self.trigger_seq)
        print(f"after epoch {epoch} model generates:")
        print("actual sequence: ", seq)
        print("generated text sequence: ", self.model.preprocessor.get_text(seq)[(0, 0)])
        