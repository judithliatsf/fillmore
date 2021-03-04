import tensorflow as tf

class MetricLearner(tf.keras.Model):
    """A learner that uses a learned distance metric to make predictions."""

    def __init__(self, embedding_func, **kwargs):
        # overwrite base class attributes
        super(MetricLearner, self).__init__(**kwargs)
        self.embedding_func = embedding_func
    
    def call(self, episode):
        support_examples = episode["support_examples"]
        query_examples = episode["query_examples"]
        support_labels_onehot = episode["support_labels_onehot"]
        support_embeddings = self.embedding_func(support_examples)
        query_embeddings = self.embedding_func(query_examples)
        query_logits = self.compute_logits_for_episode(support_embeddings, query_embeddings, support_labels_onehot)
        return query_logits

    def compute_loss(self, query_logits, query_labels_onehot):
        """generate loss for an episode

        Args:
            query_logits ([N*Q, N]): query logits from `compute_logits_for_episode`
            query_labels_onehot ([N*Q, N]): onehot label for query

        Returns:
            cost [float]: cross entropy loss averaged over all query examples
        """
        cross_entropy_loss = tf.nn.softmax_cross_entropy_with_logits(
        query_labels_onehot, query_logits)

        cost = tf.reduce_mean(cross_entropy_loss)
        return cost

    def compute_accuracy(self, query_labels_onehot, query_logits):
        """[summary]

        Args:
            query_labels_onehot ([N*Q, N]): onehot label for query
            query_logits ([N*Q, N]): query logits from `compute_logits_for_episode`

        Returns:
            accuracy [float]: accuracy averaged over all query examples
        """
        correct = tf.equal(tf.argmax(query_labels_onehot, -1), tf.argmax(query_logits, -1))
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
        return accuracy

    def _compute_prototypes(self, embeddings, labels):
        """Computes class prototypes over the last dimension of embeddings.
        Args:
            embeddings: Tensor of examples of shape [num_examples, embedding_size].
            labels: Tensor of one-hot encoded labels of shape [num_examples,
            num_classes].
        Returns:
            prototypes: Tensor of class prototypes of shape [num_classes,
            embedding_size].
        """
        labels = tf.cast(labels, tf.float32)

        # [num examples, 1, embedding size].
        embeddings = tf.expand_dims(embeddings, 1)

        # [num examples, num classes, 1].
        labels = tf.expand_dims(labels, 2)

        # Sums each class' embeddings. [num classes, embedding size].
        class_sums = tf.reduce_sum(labels * embeddings, 0)

        # The prototype of each class is the averaged embedding of its examples.
        class_num_images = tf.reduce_sum(labels, 0)  # [way].
        prototypes = class_sums / class_num_images

        return prototypes
    
    def compute_logits_for_episode(self, support_embeddings, query_embeddings, support_labels_onehot):
        """calculates the prototype network logits (before softmax layer) using the latent representation of
           support set and query set

        Args:
            support_embeddings ([N*S, D]): the latent representation of support set
            query_embeddings ([N*Q, D]): the latent representation of query set
            support_labels_onehot ([N*S, N]): the one-hot encodings of the query labels
        Returns:
            query_logits ([N*Q, N]): the query logits
        """
        prototypes = self._compute_prototypes(support_embeddings, support_labels_onehot)
        
        # compute the distance from the prototypes
        prototypes = tf.expand_dims(prototypes, axis=0)  # [1, N, D]
        query_embeddings = tf.expand_dims(query_embeddings, 1)  # [N*Q, 1, D]
        distances = tf.reduce_sum(tf.square(query_embeddings - prototypes), 2)
        return -distances