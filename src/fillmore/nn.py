import tensorflow as tf


def get_distance(emb_support, emb_query, distance_type):
    if distance_type == 'l2':
        # [1, num_support, embed_dims]
        emb_support = tf.expand_dims(emb_support, axis=0)
        # [num_query, 1, embed_dims]
        emb_query = tf.expand_dims(emb_query, axis=1)
        # [num_query, num_support]
        distance = tf.norm(emb_support - emb_query, axis=2)
    elif distance_type == 'cosine':
        emb_support = tf.nn.l2_normalize(emb_support, axis=1)
        emb_query = tf.nn.l2_normalize(emb_query, axis=1)
        # [num_query, num_support]
        distance = -1 * tf.matmul(emb_query, emb_support, transpose_b=True)
    else:
        raise ValueError('Distance must be l2 or cosine')

    return distance


def nn_eval(model, tokenizer, x, q, labels_x, labels_q):
    num_classes, num_support = x.shape
    num_queries = q.shape[1]
    x = x.reshape([num_classes*num_support, ]).tolist()
    q = q.reshape([num_classes*num_queries, ]).tolist()

    x = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=x,
        truncation=True, padding=True, max_length=model.max_seq_len, return_tensors='tf')
    q = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=q,
        truncation=True, padding=True, max_length=model.max_seq_len, return_tensors='tf')
    labels_q = tf.constant(labels_q)

    x_latent = model(x)
    q_latent = model(q)

    distance = get_distance(x_latent, q_latent, 'l2')

    _, indices = tf.nn.top_k(-distance, k=1)
    indices = tf.squeeze(indices, axis=1)

    labels_x_onehot = tf.reshape(labels_x, [-1, num_classes])
    labels_x = tf.argmax(labels_x_onehot, axis=1)
    labels_q_onehot = tf.reshape(labels_q, [-1, num_classes])
    labels_q = tf.argmax(labels_q_onehot, axis=1)

    labels_pred = tf.gather(tf.constant(labels_x), indices)
    labels_prob = tf.one_hot(labels_pred, depth=num_classes)

    bce = tf.keras.losses.BinaryCrossentropy()
    ce_loss = bce(labels_q_onehot, labels_prob)

    acc = tf.reduce_mean(
        tf.cast(tf.equal(labels_q, labels_pred), dtype=tf.float32))

    return labels_pred, ce_loss, acc
