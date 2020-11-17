import tensorflow as tf

try:
   import matplotlib.pyplot as plt
except:
    import matplotlib
    matplotlib.use('PS')
    import matplotlib.pyplot as plt

def ProtoPredict(x_latent, q_latent, num_classes, num_support):
    """
      calculates the prototype network predictions using the latent representation of x
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
    score = tf.reduce_max(prob)
#     predictions = tf.argmax(prob, 1)  # [N*Q, ]
#   predictions = tf.one_hot(predictions, num_classes)
    #############################
    return score


def get_word_importance(text, tokenizer, model, support, num_classes, num_support):
    encoded_tokens = tokenizer.encode_plus(
        text, add_special_tokens=True, return_tensors="tf")
    token_ids = list(encoded_tokens["input_ids"].numpy()[0])
    vocab_size = model.bert.embeddings.vocab_size
    x_feature = tokenizer.batch_encode_plus(
        batch_text_or_text_pairs=support,
        truncation=True, padding=True, max_length=model.max_seq_len, return_tensors='tf')

    # convert token ids to one hot. We can't differentiate wrt to int token ids hence the need for one hot representation
    token_ids_tensor = tf.constant([token_ids], dtype='int32')
    token_ids_tensor_one_hot = tf.Variable(
        tf.one_hot(token_ids_tensor, vocab_size))

    # embedding matrix
    model(model.dummy_text_inputs)
    embedding_matrix = model.bert.embeddings.word_embeddings

    with tf.GradientTape(watch_accessed_variables=False, persistent=True) as tape:
        # (i) watch input variable
        tape.watch(token_ids_tensor_one_hot)
        inputs_embeds = tf.matmul(token_ids_tensor_one_hot, embedding_matrix)
        q1_latent = model({"inputs_embeds": inputs_embeds,
                           "token_type_ids": encoded_tokens["token_type_ids"], "attention_mask": encoded_tokens["attention_mask"]})
        x_latent = model(x_feature)  # [N*S, D]
    #   q_latent = model(q) # [N*Q, D]
        score = ProtoPredict(x_latent, q1_latent,
                             num_classes, num_support)

    gradients = tape.gradient(score, token_ids_tensor_one_hot)
    gradient_non_normalized = tf.norm(gradients, axis=2)
    gradient_tensor = (
        gradient_non_normalized /
        tf.reduce_max(gradient_non_normalized)
    )

    importance = gradient_tensor[0].numpy().tolist()
    tokens = tokenizer.convert_ids_to_tokens(token_ids)
    token_types = list(encoded_tokens["token_type_ids"].numpy()[0])
    return importance, tokens, token_types


def clean_tokens(gradients, tokens, token_types):
    """
    Clean the tokens and gradients gradients
    Remove "[CLS]","[CLR]", "[SEP]" tokens
    Reduce (mean) gradients values for tokens that are split ##
    """
    token_holder = []
    token_type_holder = []
    gradient_holder = []
    i = 0
    while i < len(tokens):
        if (tokens[i] not in ["[CLS]", "[CLR]", "[SEP]"]):
            token = tokens[i]
            conn = gradients[i]
            token_type = token_types[i]
            if i < len(tokens)-1:
                if tokens[i+1][0:2] == "##":
                    token = tokens[i]
                    conn = gradients[i]
                    j = 1
                    while i < len(tokens)-1 and tokens[i+1][0:2] == "##":
                        i += 1
                        token += tokens[i][2:]
                        conn += gradients[i]
                        j += 1
                    conn = conn / j
            token_holder.append(token)
            token_type_holder.append(token_type)
            gradient_holder.append(conn)
        i += 1
    return gradient_holder, token_holder, token_type_holder


def plot_gradients(tokens, token_types, gradients, title):
    """ Plot  explanations
    """
    plt.figure(figsize=(21, 3))
    xvals = [x + str(i) for i, x in enumerate(tokens)]
    colors = [(0, 0, 1, c) for c, t in zip(gradients, token_types)]
    edgecolors = ["black" if t == 0 else (
        0, 0, 1, c) for c, t in zip(gradients, token_types)]
    # colors =  [  ("r" if t==0 else "b")  for c,t in zip(gradients, token_types) ]
    plt.tick_params(axis='both', which='minor', labelsize=29)
    p = plt.bar(xvals, gradients, color=colors,
                linewidth=1, edgecolor=edgecolors)
    plt.title(title)
    p = plt.xticks(ticks=[i for i in range(len(tokens))],
                   labels=tokens, fontsize=12, rotation=90)
    
if __name__ == "__main__":
    from transformers import AutoConfig, BertTokenizer
    from fillmore.bert_model import BertTextEncoder
    import os
    folder = "logs/proto/cls_5.eps_2.k_shot_10.n_query_10"
    config = AutoConfig.from_pretrained(os.path.join(folder, "config.json"))
    model = BertTextEncoder(config)
    tokenizer = BertTokenizer.from_pretrained(
                'bert-base-uncased')

    support = ['am i allowed to put in a pto request for now to april',
                'can you set up a pto request for me for march 20th to april 12th',
                'can you put in a pto request for the 5th of june to june 8th',
                'i need a pto request for the dates jan 15th to jan 20th',
                'i need to put in a pto request from the 8th to the 17th of january']
    text = "i want to schedule a pto request on march 1-2"
    num_classes = 5
    num_support = 1
    # text = "i need a pto request put in for the weekend of june 1st to june 2nd"
    importance, tokens, token_types = get_word_importance(text, tokenizer, model, support, num_classes, num_support)
    importance, tokens, token_types = clean_tokens(importance, tokens, token_types)
    plot_gradients(tokens, token_types, importance, 'Importance for "{}" '.format(text))