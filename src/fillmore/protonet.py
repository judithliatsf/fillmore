# models/ProtoNet
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers


class ProtoNet(tf.keras.Model):

    def __init__(self, num_filters, latent_dim):
        super(ProtoNet, self).__init__()
        self.num_filters = num_filters
        self.latent_dim = latent_dim
        num_filter_list = self.num_filters + [latent_dim]
        self.convs = []
        for i, num_filter in enumerate(num_filter_list):
            block_parts = [
                layers.Conv2D(
                    filters=num_filter,
                    kernel_size=3,
                    padding='SAME',
                    activation='linear'),
            ]

            block_parts += [layers.BatchNormalization()]
            block_parts += [layers.Activation('relu')]
            block_parts += [layers.MaxPool2D()]
            block = tf.keras.Sequential(block_parts, name='conv_block_%d' % i)
            self.__setattr__("conv%d" % i, block)
            self.convs.append(block)
        self.flatten = tf.keras.layers.Flatten()

    def call(self, inp):
        out = inp
        for conv in self.convs:
            out = conv(out)
        out = self.flatten(out)
        return out


def ProtoLoss(x_latent, q_latent, labels_onehot, num_classes, num_support, num_queries, config):
    """
      calculates the prototype network loss using the latent representation of x
      and the latent representation of the query set
      Args:
        x_latent: latent representation of supports with shape [N*S, D], where D is the latent dimension
        q_latent: latent representation of queries with shape [N*Q, D], where D is the latent dimension
        labels_onehot: one-hot encodings of the labels of the queries with shape [N, Q, N]
        num_classes: number of classes (N) for classification
        num_support: number of examples (S) in the support set
        num_queries: number of examples (Q) in the query set
      Returns:
        ce_loss: the cross entropy loss between the predicted labels and true labels
        acc: the accuracy of classification on the queries
    """
    #############################
    #### YOUR CODE GOES HERE ####

    # compute the prototypes
    _, D = tf.shape(x_latent)
    x = tf.reshape(x_latent, [num_classes, num_support, D])
    prototypes = tf.reduce_mean(x, 1)

    # compute the distance from the prototypes
    prototypes = tf.expand_dims(prototypes, axis=0)  # [1, N, D]
    q_latent = tf.expand_dims(q_latent, 1)  # [N*Q, 1, D]
    distances = tf.reduce_sum(tf.square(q_latent - prototypes), 2)

    # compute cross entropy loss
    prob = tf.nn.softmax(-distances)
    predictions = tf.argmax(prob, 1)  # [N*Q, ]
#   predictions = tf.one_hot(predictions, num_classes)

    # note - additional steps are needed!
    # return the cross-entropy loss and accuracy
    labels_onehot = tf.reshape(labels_onehot, [-1, num_classes])
    labels = tf.argmax(labels_onehot, axis=1)

    bce = tf.keras.losses.BinaryCrossentropy(label_smoothing=config.label_smoothing)
    ce_loss = bce(labels_onehot, prob)

    acc = tf.reduce_mean(
        tf.cast(tf.equal(labels, predictions), dtype=tf.float32))
    #############################
    return ce_loss, acc
