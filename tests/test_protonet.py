import tensorflow as tf
from fillmore.protonet import ProtoLoss

labels = [[0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]]
preds  = [[0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
          [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1]]
labels1 = tf.argmax(labels, axis=1)
preds1 = tf.argmax(preds, axis=1)
acc = tf.reduce_mean(tf.cast(tf.equal(labels1, preds1), dtype=tf.float32))
print(acc)

def _compute_prototypes(embeddings, labels):
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

# inputs
x_latent = tf.constant([[[1.0, 1.0], [0.0, 1.0], [0.5, 0.5]], [[1.0, -1.0], [0.0, -1.0], [0.5, -0.5]]]) # [N*S, D]
x_latent = tf.reshape(x_latent, [2*3, 2])
q_latent = tf.constant([[[0.0, 0.5]], [[0.0, -0.5]]])
q_latent = tf.reshape(q_latent, [2*1, 2])
num_classes = 2
num_support = 3
num_queries = 1
labels_onehot = tf.constant([[[1., 0.]], [[1., 0.]]]) # [N, Q, N]
x_labels_onehot = tf.constant([[[1., 0.], [1., 0.], [1., 0.]], [[0., 1.], [0., 1.], [0., 1.]]])
x_labels_onehot = tf.reshape(x_labels_onehot, [2*3, 2])

loss, acc = ProtoLoss(x_latent, q_latent, labels_onehot, num_classes, num_support, num_queries)
print("loss: {}".format(loss))
print("acc: {}".format(acc))

# tests
_, D = tf.shape(x_latent)
x = tf.reshape(x_latent, [num_classes, num_support, D])
prototypes = tf.reduce_mean(x, 1)
print(prototypes)
# assert prototypes == tf.constant(
# [[ 0.5,0.8333333],
#  [ 0.5,-0.8333333]], dtype=tf.float32)

# compute the distance from the prototypes
prototypes = tf.expand_dims(prototypes, axis=0) # [1, N, D]
q_latent = tf.expand_dims(q_latent, 1) #[N*Q, 1, D]
distances = tf.reduce_sum(tf.square(q_latent - prototypes), 2)
print("distances")
print(distances.shape) # [N*Q, N]
print(distances)

# compute cross entropy loss
prob = tf.nn.softmax(-distances)
print("prob")
print(prob)

print("predictions")
predictions = tf.argmax(prob, 1) # [N*Q, ]
# predictions = tf.one_hot(predictions, num_classes)
print(predictions.shape)
print(predictions)

# note - additional steps are needed!
# return the cross-entropy loss and accuracy
labels_onehot = tf.reshape(labels_onehot, [-1, num_classes])
labels_final = tf.argmax(labels_onehot, axis=1)
print("true labels")
print(labels_onehot)
print(labels_final)

bce = tf.keras.losses.BinaryCrossentropy()
ce_loss = bce(labels_onehot, prob)
print(ce_loss)

acc = tf.reduce_mean(tf.cast(tf.equal(labels_final, predictions), dtype=tf.float32))
print(acc)