import tensorflow as tf

# custom loss for model
class MaskedLoss(tf.keras.losses.Loss):
    def __init__(self):
        super().__init__()
        self.name="masked-loss"
        self.loss_function = tf.keras.losses.SparseCategoricalCrossentropy(reduction='none')

    def call(self, y_true, y_pred):
        mask = tf.cast(tf.math.logical_not(tf.math.equal(y_true, 0)), dtype=tf.float32)
        loss = self.loss_function(y_true, y_pred)
        loss *= mask
        return tf.reduce_sum(loss)/tf.reduce_sum(mask)


# custom Accuracy for model
class MaskedAccuracy(tf.keras.metrics.Metric):
    def __init__(self):
        super().__init__(name="Masked-Accuracy")
        self.accuracy = self.add_weight(name='tp', initializer='zeros')

    def update_state(self, y_true, y_pred, sample_weight=None):
        mask = tf.cast(tf.math.logical_not(tf.math.equal(y_true, 0)), dtype=tf.float32)
        accuracies =  tf.cast(tf.math.equal(tf.cast(y_true, dtype=tf.int64), tf.argmax(y_pred, axis=-1)), dtype=tf.float32)
        accuracies *= mask
        
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, self.dtype)
            sample_weight = tf.broadcast_to(sample_weight, accuracies.shape)
            accuracies = tf.multiply(accuracies, sample_weight)

        self.accuracy = self.accuracy.assign(tf.reduce_sum(accuracies)/tf.reduce_sum(mask))
    
    def result(self):
        return self.accuracy
