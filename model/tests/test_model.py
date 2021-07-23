
## Do not touch these codes
## Essential for unit testing

import tensorflow as tf

from model.model import Block, Poet
from model.metrics import MaskedAccuracy, MaskedLoss
from model.callbacks import SeqGenerateCallback
from model.preprocess import Preprocessor


## test code for Block
layer = Block(d_model=8, dff=256, heads=4)
mask = tf.keras.Input(shape=[4, 4])
source = tf.keras.Input(shape=[4, 100])
outputs = layer(inputs=source, mask=mask)
print(outputs.shape)


## test code for Poet
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

model = Poet(preprocessor=preprocessor, num_blocks=1, d_model=256, dff=512, heads=8, embedding_dims=100)
model.compile(loss=MaskedLoss(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=[MaskedAccuracy()])



string_inputs = [["i will love you"], ["i like you"]]
trigger_inputs = [["i"]]
vocab = preprocessor.build_vocab(string_inputs)
print(vocab)
preprocess_inputs = preprocessor(string_inputs)
trigger_inputs = preprocessor(trigger_inputs)

history = model.fit(x=preprocess_inputs, y=preprocess_inputs, batch_size=1, epochs=4, callbacks=[SeqGenerateCallback(trigger_inputs)])