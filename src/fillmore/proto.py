import numpy as np
import tensorflow as tf
from fillmore.dataset.loader import load_dataset
from fillmore.dataset.data_loader import TextDataLoader

def proto_net_train():
    pass

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

def create_data_loaders(config):

    # load IntentExamples for target task
    data = load_dataset(config)

    # create data_loader for each SPLIT
    data_loaders = {
        "meta_train": TextDataLoader(
            data["meta_train"], 
            config.k_shot, config.n_query, config.n_way,
            seed=config.seed, task=config.dataset
        ),
        "meta_val": TextDataLoader(
            data["meta_val"], 
            config.k_shot, config.n_query, config.n_way,
            seed=config.seed, task=config.dataset
        ),
        "meta_test": TextDataLoader(
            data["meta_test"], 
            config.k_meta_test_shot, config.n_meta_test_query, config.n_meta_test_way,
            seed=config.seed, task=config.dataset
        )
    }

    if config.smlmt:
        data_loaders['smlmt_train'] = TextDataLoader(
            data["smlmt_train"],
            config.smlmt_k_shot, config.smlmt_n_query, config.n_way,
            seed=config.seed, task=config.dataset
        )

    if config.oos:
        data_loaders["oos_val"] = TextDataLoader(
            data["oos_val"],
            config.k_shot, config.n_query, 1,
            seed=config.seed, task=config.dataset
        )
        data_loaders["oos_test"] = TextDataLoader(
            data["oos_test"],
            config.k_meta_test_shot, config.n_meta_test_query, 1,
            seed=config.seed, task=config.dataset
        )
    
    return data_loaders

if __name__ == "__main__":
    from transformers import BertConfig
    
    config=BertConfig.from_dict({
        "dataset": "clinc150c",
        "data_path": "data/clinc150.json",
        "num_examples_from_class_train": 20,
        "num_examples_from_class_valid": 50,
        "num_examples_from_class_test": 50,
        "n_way": 5,
        "k_shot": 10,
        "n_query": 10,
        "n_meta_test_way": 5,
        "k_meta_test_shot": 10,
        "n_meta_test_query": 10,
        "oos": True,
        "oos_data_path": "data/clinc150_oos.json", 
        "smlmt": True,
        "smlmt_ratio": 0.6,
        "smlmt_k_shot": 15,
        "smlmt_n_query": 10,
        "seed": 1234
    })

    # create DataLoader for SPLIT
    data_loaders = create_data_loaders(config)

    # # load episode
    # SPLIT="meta_val"
    # batch_size = 2
    # episodes = data_loaders[SPLIT].sample_episodes(batch_size)

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
    config.n_episodes = 1 # number of training episodes per epoch
    config.n_val_episodes = 2 # number of validation episodes
    config.n_epochs = 1 # number of training epochs

    exp_string = 'cls_'+str(config.n_way)+'.eps_'+str(config.n_episodes) + \
        '.k_shot_' + str(config.k_shot) + '.n_query_' + str(config.n_query)
    
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=config.learning_rate,
        epsilon=config.epsilon,
        clipnorm=config.clipnorm
    )
    for ep in range(config.n_epochs):
        for epi in range(config.n_episodes):
            episode = data_loaders["meta_train"].sample_episodes(1)[0]
            loss, grad = proto_net_train_step(model, optimizer, episode, config)    
            episode_stats = proto_net_eval_step(model, episode, config)
            print("loss: {}".format(loss))
            print("acc: {}".format(episode_stats["acc"]))

            episode = data_loaders["smlmt_train"].sample_episodes(1)[0]
            loss, grad = proto_net_train_step(model, optimizer, episode, config)    
            episode_stats = proto_net_eval_step(model, episode, config)
            print("loss: {}".format(loss))
            print("acc: {}".format(episode_stats["acc"]))
            
            # load episodes for evaluation
            stats = proto_net_eval(model, data_loaders, config.n_val_episodes, "meta_val", config)
            print("val acc: {}".format(stats["acc"]))