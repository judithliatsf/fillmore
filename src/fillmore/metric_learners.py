import tensorflow as tf

class MetricLearner(tf.keras.Model):
    """A learner that uses a learned distance metric to make predictions."""
    def call(self):
        pass

    def compute_loss(self):
        pass

    def compute_logits_for_episode(self):
        pass

    def compute_accuracy(self):
        pass

    def embedding_func(self):
        pass