import tensorflow as tf
from model.metrics import MaskedLoss, MaskedAccuracy

# test code
preprocess_inputs = tf.constant([[1,2,3,2,4], [2,5,0,0,0]], dtype=tf.float16)
print(preprocess_inputs)
targets = tf.constant([[1,2,3,2,4], [2,5,0,0,0]], dtype=tf.float16)

loss = MaskedLoss().call(preprocess_inputs, targets)
print("loss shape:", loss.shape)

accuracy = MaskedAccuracy()
accuracy.update_state(preprocess_inputs, targets)
print("accuracy:", accuracy.result().numpy())