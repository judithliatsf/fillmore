import tensorflow as tf

class MetricLearner(tf.keras.Model):
    """A learner that uses a learned distance metric to make predictions."""

    def __init__(self, embedding_func, **kwargs):
        # overwrite base class attributes
        super(MetricLearner, self).__init__(**kwargs)
        self.embedding_func = embedding_func
    
    def call(self, episode, query_examples=[]):
        support_examples = episode["support_examples"]
        if not query_examples:
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
        tf.stop_gradient(query_labels_onehot), query_logits)

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

class KNNLearner(tf.keras.Model):
    """A learner that uses a learned distance metric to make predictions."""

    def __init__(self, embedding_func, distance_type='l2', **kwargs):
        # overwrite base class attributes
        super(KNNLearner, self).__init__(**kwargs)
        self.embedding_func = embedding_func
        self.distance_type = distance_type
    
    def call(self, episode, query_examples=[]):
        support_examples = episode["support_examples"]
        if not query_examples:
            query_examples = episode["query_examples"]
        support_labels_onehot = episode["support_labels_onehot"]
        support_embeddings = self.embedding_func(support_examples)
        query_embeddings = self.embedding_func(query_examples)
        query_logits = self.compute_logits_for_episode(support_embeddings, query_embeddings, support_labels_onehot)
        return query_logits

    def compute_loss(self, query_prob, query_labels_onehot):
        """generate loss for an episode

        Args:
            query_prob ([N*Q, N]): query probability from `compute_logits_for_episode`
            query_labels_onehot ([N*Q, N]): onehot label for query
            support_labels_onehot ([N*S, N]): onehot label for support

        Returns:
            cost [float]: cross entropy loss averaged over all query examples
        """

#         bce = tf.keras.losses.BinaryCrossentropy()
        bce = tf.keras.losses.BinaryCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.NONE)
        cross_entropy_loss = bce(query_labels_onehot, query_prob)

        cost = tf.reduce_mean(cross_entropy_loss)
        return cost

    def compute_accuracy(self, query_labels_onehot, query_prob):
        """[summary]

        Args:
            query_labels_onehot ([N*Q, N]): onehot label for query
            query_prob ([N*Q, N]): query probability from `compute_logits_for_episode`

        Returns:
            accuracy [float]: accuracy averaged over all query examples
        """
        correct = tf.equal(tf.argmax(query_labels_onehot, -1), tf.argmax(query_prob, -1), -1)
        correct = tf.cast(correct, tf.float32)
        accuracy = tf.reduce_mean(correct)
        return accuracy
      
    def compute_logits_for_episode(self, support_embeddings, query_embeddings, support_labels_onehot):
        """calculates the nearest neighbors prediction using the latent representation of
           support set and query set

        Args:
            support_embeddings ([N*S, D]): the latent representation of support set
            query_embeddings ([N*Q, D]): the latent representation of query set
            support_labels_onehot ([N*S, N]): the one-hot encodings of the query labels
        Returns:
            query_logits ([N*Q, N]): the query logits
        """
        if self.distance_type == 'l2':
            # [1, num_support, embed_dims]
            emb_support = tf.expand_dims(support_embeddings, axis=0)
            # [num_query, 1, embed_dims]
            emb_query = tf.expand_dims(query_embeddings, axis=1)
            # [num_query, num_support]
            distance = tf.norm(emb_support - emb_query, axis=2)
        elif self.distance_type == 'cosine':
            emb_support = tf.nn.l2_normalize(emb_support, axis=1)
            emb_query = tf.nn.l2_normalize(emb_query, axis=1)
            # [num_query, num_support]
            distance = -1 * tf.matmul(emb_query, emb_support, transpose_b=True)
        else:
            raise ValueError('Distance must be l2 or cosine')

        _, indices = tf.nn.top_k(-distance, k=1)
        indices = tf.squeeze(indices, axis=1)

#         labels_x_onehot = tf.reshape(labels_x, [-1, num_classes])
        labels_x = tf.argmax(support_labels_onehot, axis=1)
#         labels_q_onehot = tf.reshape(labels_q, [-1, num_classes])
#         labels_q = tf.argmax(labels_q_onehot, axis=1)

        num_classes = support_labels_onehot.shape[1]
        labels_pred = tf.gather(tf.constant(labels_x), indices)
        labels_prob = tf.one_hot(labels_pred, depth=num_classes)
        return labels_prob
