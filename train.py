import tensorflow as tf
import numpy as np
import pandas as pd
import seaborn as sns

from model.preprocess import Preprocessor
from model.callbacks import SeqGenerateCallback
from model.model import Poet
from model.metrics import MaskedAccuracy, MaskedLoss



MAX_VOCAB_SIZE = 10000
EMBEDDING_DIM = 200
DFF = 512
D_MODEL = 256
MAX_SEQ_LEN = 10



data = pd.read_csv("data.tsv", sep="\t")


data.head()

train_data = data.iloc[:-5, :]
val_data = data.iloc[-5:, :]

train_data.shape

preprocessor = Preprocessor(vocab_size=1000, seq_len=10)

vocab = preprocessor.build_vocab(train_data)
x_train_seq, y_train_seq = preprocessor(train_data, training=True)
x_val_seq, y_val_seq = preprocessor(val_data, training=True)

x_train_seq.shape, y_train_seq.shape, x_val_seq.shape, y_val_seq.shape


x_train_seq[:3]

print("few vocab tokens:", vocab[:10])


print("Vocab Size: ", preprocessor.vocab_size)


model = Poet(preprocessor=preprocessor, num_blocks=1, d_model=256, dff=512, heads=8, embedding_dims=100)
model.compile(loss=MaskedLoss(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=[MaskedAccuracy()])


trigger_inputs = [["मै"]]
trigger_inputs = preprocessor(trigger_inputs, training=False)
generator_callback = SeqGenerateCallback(trigger_inputs)


history = model.fit(x=x_train_seq, y=y_train_seq, batch_size=5, epochs=10, callbacks=[generator_callback])


