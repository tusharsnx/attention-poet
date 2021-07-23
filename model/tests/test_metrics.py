import tensorflow as tf
from model.metrics import MaskedLoss, MaskedAccuracy


# test code
y_pred = tf.constant([[[0.1, 0.8, 0.1], [0.8, 0.1, 0.1]]], dtype=tf.float32)
print(y_pred.shape)
y_true = tf.constant([[1, 0]], dtype=tf.float16)

loss = MaskedLoss().call(y_true=y_true, y_pred=y_pred)
print("loss:", loss.numpy())

accuracy = MaskedAccuracy()
accuracy.update_state(y_true=y_true, y_pred=y_pred)
print("accuracy:", accuracy.result().numpy())