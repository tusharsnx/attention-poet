from tensorflow.keras.callbacks import Callback

## callback to generate seq at end of each epoch
class SeqGenerateCallback(Callback):
    def __init__(self, trigger_seq):
        super().__init__()
        self.trigger_seq = trigger_seq
        
    def on_epoch_end(self, epoch, logs=None):
        text, seq = self.model.generate(self.trigger_seq, return_seq=True)
        print(f"after epoch {epoch+1} model generates: ")
        print("actual sequence: ", seq)
        print("generated text sequence: ", text)
        