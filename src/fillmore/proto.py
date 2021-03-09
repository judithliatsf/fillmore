import numpy as np
from numpy.lib.function_base import gradient
import tensorflow as tf

def proto_net_train_step(model, opt, episode, config):
    """Update model by backpropagating training loss per episode

    Args:
        model (Learner): Learner for forward pass and compute loss
        opt (tf.keras.Optimizer): Optimizer, e.g., Adam
        episode ([type]): An episode contains support and query sets
        config ([type]): A config access by `config.attribute`

    Returns:
        [type]: [description]
    """
    with tf.GradientTape() as tape:
        query_logits = model(episode)
        query_labels_onehot = episode["query_labels_onehot"]
        loss = model.compute_loss(query_logits, query_labels_onehot)

    gradients = tape.gradient(loss, model.trainable_variables)
    opt.apply_gradients(zip(gradients, model.trainable_variables))
    return loss, gradients
    
def proto_net_eval(model, data_loaders, num_episodes, split, config):
    """Compute accuracy and other statistis averaged over a batch of episodes

    Args:
        model (Learner): A few-shot learner
        data_loaders (dict): A dictionary of data_loader for each split
        num_episodes (int): number of episodes used for evaluation 
        split (string): e.g., "meta_val"
        config (dict): A config access by `config.attribute`

    Returns:
        batch_stats: statistics average over a batch of episodes
    """
    data_loader = data_loaders[split]

    meta_eval_accuracies = []

    for episode in data_loader.sample_episodes(num_episodes):
        episode_stats = proto_net_eval_step(model, episode, config)
        meta_eval_accuracies.append(episode_stats["acc"])
    
    avg_acc = np.mean(meta_eval_accuracies)
    stds = np.std(meta_eval_accuracies)

    return {"acc": avg_acc, "stds": stds}

def proto_net_eval_step(model, episode, config):
    """Compute query accuracy for an episode given a few-shot learner

    Args:
        model (Learner): A learner compute query accuracy for an episode
        episode (Dict): An episode contains support and query sets
        config (Dict): A config access by `config.attribute`

    Returns:
        episode_stats: statistics of one episode
    """
    query_logits= model(episode)
    query_labels_onehot = episode['query_labels_onehot']
    acc = model.compute_accuracy(query_labels_onehot, query_logits)
    episode_stats = {
        "acc": acc
    }
    return episode_stats


if __name__ == "__main__":
    from transformers import BertConfig
    from fillmore.dataset.data_loader import TextDataLoader
    from fillmore.dataset.loader import load_dataset
    config=BertConfig.from_dict({
        "dataset": "clinc150c",
        "data_path": "data/clinc150.json",
        "num_examples_from_class_train": 20,
        "num_examples_from_class_valid": 50,
        "num_examples_from_class_test": 50,
        "oos": True,
        "oos_data_path": "data/clinc150_oos.json", 
        "n_way": 5,
        "k_shot": 10,
        "n_query": 10,
        "n_meta_test_way": 5,
        "k_meta_test_shot": 10,
        "n_meta_test_query": 10
    })

    if config.oos:
        train_data_by_class, val_data_by_class, test_data_by_class, oos_val_data_by_class, oos_test_data_by_class=load_dataset(config)
    else:
        train_data_by_class, val_data_by_class, test_data_by_class=load_dataset(config)
    
    # create DataLoader for SPLIT
    data_loaders={"meta_train": TextDataLoader(
            train_data_by_class, 
            config.k_shot, config.n_query, config.n_way,
            seed=1234, task=config.dataset
        ),
        "meta_val": TextDataLoader(
            train_data_by_class, 
            config.k_shot, config.n_query, config.n_way,
            seed=1234, task=config.dataset
        ),
        "meta_test": TextDataLoader(
            train_data_by_class, 
            config.k_meta_test_shot, config.n_meta_test_query, config.n_meta_test_way,
            seed=1234, task=config.dataset
        ),
        "oos_val": TextDataLoader(
            oos_val_data_by_class,
            config.k_shot, config.n_query, 1,
            seed=1234, task=config.dataset
        ),
        "oos_test": TextDataLoader(
            oos_test_data_by_class,
            config.k_meta_test_shot, config.n_meta_test_query, 1,
            seed=1234, task=config.dataset
        )
        }

    # load episode
    SPLIT="meta_val"
    batch_size = 2
    episodes = data_loaders[SPLIT].sample_episodes(batch_size)

    # load model
    from fillmore.metric_learners import MetricLearner
    from fillmore.bert_model import BertTextEncoderWrapper
    config.label_smoothing = 0.0
    config.max_seq_len = 32
    embedding_func = BertTextEncoderWrapper(config)
    model = MetricLearner(embedding_func)
    
    # train
    config.learning_rate = 1e-5
    config.epsilon = 1e-8
    config.clipnorm = 1.0
    train_episodes = data_loaders["meta_train"].sample_episodes(batch_size)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate,
        epsilon=config.epsilon,
        clipnorm=config.clipnorm
    )
    for episode in train_episodes:
        loss, grad = proto_net_train_step(model, optimizer, episode, config)    
        episode_stats = proto_net_eval_step(model, episode, config)
        print("loss: {}".format(loss))
        print("acc: {}".format(episode_stats["acc"]))
    
    # load episodes for evaluation
    # stats = proto_net_eval(model, data_loaders, batch_size, SPLIT, config)
    # print(stats)
    # for episode in episodes:
    #     episode_stats = proto_net_eval_step(model, episode, config)
    #     print(episode_stats["acc"])