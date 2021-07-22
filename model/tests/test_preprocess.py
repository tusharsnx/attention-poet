from model.preprocess import Preprocessor

## test code
preprocessor = Preprocessor(vocab_size=100, seq_len=10)
inputs = [["जैसा "], ["i am fine, what about you. i mean ? "]]
vocab = preprocessor.build_vocab(inputs)
print(vocab)
outputs = preprocessor(inputs)
print(outputs)
print(preprocessor.get_text(outputs.numpy()))