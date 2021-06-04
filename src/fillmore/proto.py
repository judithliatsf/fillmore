import numpy as np
import tensorflow as tf
from fillmore.dataset.loader import load_dataset
from fillmore.dataset.data_loader import TextDataLoader
from fillmore.utils import count_class
from fillmore.metrics import *
from tqdm import tqdm
import mlflow

def proto_net_train(config, model, optimizer, retrain=False, ckpt_manager=None):

    # train
    if retrain:
        status = checkpoint.restore(manager.latest_checkpoint)
        print("Load model weights from checkpoints ", manager.latest_checkpoint)
    
    # start training
    print("start training...")
    seen_task_stats = proto_net_eval(model, data_loaders["meta_train"], config.n_meta_val_episodes, config)
    unseen_task_stats = proto_net_eval(model, data_loaders["meta_val"], config.n_meta_val_episodes, config)
    mlflow.log_metric("meta-val-loss-seen-task", seen_task_stats["loss"], step = optimizer.iterations.numpy())
    mlflow.log_metric("meta-val-acc-seen-task", seen_task_stats["acc"], step = optimizer.iterations.numpy())
    mlflow.log_metric("meta-val-loss-unseen-task", unseen_task_stats["loss"], step = optimizer.iterations.numpy())
    mlflow.log_metric("meta-val-acc-unseen-task", unseen_task_stats["acc"], step = optimizer.iterations.numpy())
    
    # add metric to record classes being sampled
    train_class_counter = {}
    smlmt_class_counter = {}
    best_val_acc_unseen = 0.0
    patience = 0.0
    prev_val_acc_unseen = 0.0
    min_delta = 0.01
    max_patience = 5

    for ep in range(config.n_epochs):

        print("epoch: {}".format(ep))
        epoch_train_loss = []
        epoch_train_acc = []

        for epi in tqdm(range(config.n_meta_train_episodes), total=(config.n_epochs*config.n_meta_train_episodes)):
            
            # set learning rate
            global_step = optimizer.iterations.numpy()
            lr = optimizer.learning_rate(tf.cast(optimizer.iterations, tf.float32))
            mlflow.log_metric("meta-train-lr", lr.numpy(), step=global_step)
           
            if not config.smlmt:
                episode = data_loaders["meta_train"].sample_episodes(1)[0]
                count_class(episode, train_class_counter)
            else:
                lucky_number = np.random.random_sample()
                if lucky_number < config.smlmt_ratio:
                    episode = data_loaders["smlmt_train"].sample_episodes(1)[0]
                    count_class(episode, smlmt_class_counter)
                else:
                    episode = data_loaders["meta_train"].sample_episodes(1)[0]
                    count_class(episode, train_class_counter)
            
            loss, grad = proto_net_train_step(model, optimizer, episode, config)    
            episode_stats = proto_net_eval_step(model, episode, config)
            epoch_train_loss.append(episode_stats["loss"])
            epoch_train_acc.append(episode_stats["acc"])
            
        # training batch(epoch) loss and acc
        ep_train_loss = np.mean(epoch_train_loss)
        ep_train_acc = np.mean(epoch_train_acc)
        print("training loss: {}".format(ep_train_loss))
        print("training acc: {}".format(ep_train_acc))
        mlflow.log_metric("meta-train-loss", ep_train_loss, step = optimizer.iterations.numpy())
        mlflow.log_metric("meta-train-acc", ep_train_acc, step = optimizer.iterations.numpy())

        # meta validation on seen task
#         seen_task_stats = proto_net_eval(model, data_loaders["meta_train"], config.n_meta_val_episodes, config)
#         print("seen task val loss: {}".format(seen_task_stats["loss"]))
#         print("seen task val acc: {}".format(seen_task_stats["acc"]))
#         mlflow.log_metric("meta-val-loss-seen-task", seen_task_stats["loss"], step = optimizer.iterations.numpy())
#         mlflow.log_metric("meta-val-acc-seen-task", seen_task_stats["acc"], step = optimizer.iterations.numpy())

       # meta validation on unseen task
        unseen_task_stats = proto_net_eval(model, data_loaders["meta_val"], config.n_meta_val_episodes, config)
        print("meta-val-acc-unseen-task: {}".format(unseen_task_stats["acc"]))
        print("meta-val-loss-unseen-task: {}".format(unseen_task_stats["loss"]))
        mlflow.log_metric("meta-val-loss-unseen-task", unseen_task_stats["loss"], step = optimizer.iterations.numpy())
        mlflow.log_metric("meta-val-acc-unseen-task", unseen_task_stats["acc"], step = optimizer.iterations.numpy())

        # update best validation accurace
        if unseen_task_stats["acc"] > best_val_acc_unseen:
          best_val_acc_unseen = unseen_task_stats["acc"]
          ckpt_manager.save()
        
        # check improvements of validation acc
        if unseen_task_stats["acc"] - prev_val_acc_unseen > min_delta:
          patience = 0
        else:
          patience = patience + 1
        
        # save model at the end of every epoch
        if patience > max_patience:
          print("early stop at episodes: {}".format(optimizer.iterations.numpy()))
          break

    config.save_pretrained(config.run_dir)
    mlflow.log_artifacts(config.run_dir, artifact_path="logs")
    return train_class_counter, smlmt_class_counter        

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
    
def proto_net_eval(model, data_loader, num_episodes, config):
    """Compute accuracy and other statistis averaged over a batch of episodes

    Args:
        model (Learner): A few-shot learner
        data_loaders (TextDataLoader): A episode loader
        num_episodes (int): number of episodes used for evaluation 
        split (string): e.g., "meta_val"
        config (dict): A config access by `config.attribute`

    Returns:
        batch_stats: statistics average over a batch of episodes
    """
    meta_eval_accuracies = []
    losses = []

    for episode in data_loader.sample_episodes(num_episodes):
        episode_stats = proto_net_eval_step(model, episode, config)
        meta_eval_accuracies.append(episode_stats["acc"])
        losses.append(episode_stats["loss"])
    
    avg_acc = np.mean(meta_eval_accuracies)
    stds = np.std(meta_eval_accuracies)
    loss = np.mean(losses)
    return {"acc": avg_acc, "stds": stds, "loss": loss}

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
    loss = model.compute_loss(query_logits, query_labels_onehot)
    episode_stats = {
        "acc": acc.numpy(),
        "loss": loss.numpy()
    }
    return episode_stats

def evaluate_oos_f1(model, data_loader, num_episodes, config):
    thresholds = np.arange(0.0, 1.0, 0.1)
    in_domain_correct_all = []
    oos_correct_all = []
    oos_output_all = []

    for i in tqdm(range(num_episodes)):
        oos_episode = data_loader.sample_episodes(1)[0]
        in_domain_correct, oos_output, oos_correct = evaluate_oos_f1_step(model, oos_episode, thresholds)
        in_domain_correct_all.append(in_domain_correct.numpy())
        oos_correct_all.append(oos_correct.numpy())
        oos_output_all.append(oos_output.numpy())

    oos_stats = oos_f1_batch(in_domain_correct_all, oos_correct_all, oos_output_all)
    stats = select_oos_threshold(oos_stats, thresholds)
    return stats

def evaluate_oos_f1_step(model, oos_episode, thresholds):

    oos_prob = model.predict_prob(oos_episode, oos_episode["oos_examples"])
    query_prob = model.predict_prob(oos_episode, oos_episode["query_examples"])
    query_labels_onehot = oos_episode["query_labels_onehot"]

    in_domain_correct, oos_output = in_domain_stats_episode(query_prob, query_labels_onehot, thresholds)

    oos_correct = oos_stats_episode(oos_prob, thresholds)

    return in_domain_correct, oos_output, oos_correct

def analyze_oos_f1_step(model, oos_episode, threshold):

    oos_prob = model.predict_prob(oos_episode, oos_episode["oos_examples"])
    query_prob = model.predict_prob(oos_episode, oos_episode["query_examples"])
    query_labels_onehot = oos_episode["query_labels_onehot"]

    in_domain_correct, oos_output = in_domain_stats_episode(query_prob, query_labels_onehot, [threshold])

    oos_correct = oos_stats_episode(oos_prob, [threshold])

    return in_domain_correct, oos_output, oos_correct

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
        # include oos samples as an additional class in meta_val or meta_test
        data_loaders["oos_val"] = TextDataLoader(
            data["meta_val"],
            config.k_shot, config.n_query, config.n_way,
            seed=config.seed, task=config.dataset,
            oos=config.oos, oos_data_by_class=data["oos_val"]
        )
        data_loaders["oos_test"] = TextDataLoader(
            data["meta_test"],
            config.k_meta_test_shot, config.n_meta_test_query, config.n_meta_test_way,
            seed=config.seed, task=config.dataset,
            oos=config.oos, oos_data_by_class=data["oos_test"]
        )
    
    return data_loaders

if __name__ == "__main__":
    from transformers import BertConfig
    import os
    from fillmore.utils import WarmupLRScheduler

    meta_train = True
    meta_test = True
    retrain = False
    # checkpoint_name = None

    config=BertConfig.from_dict({
        "experiment_dir": "logs/proto1",
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
    
    # train config
    config.learning_rate = 3e-3
    config.warm_up_prop = 0.1
    config.epsilon = 1e-8
    config.clipnorm = 1.0
    config.n_meta_train_episodes = 1 # number of training episodes per epoch
    config.n_meta_val_episodes = 2 # number of validation episodes
    config.n_meta_test_episodes = 3
    config.n_epochs = 2 # number of training epochs
    
    # checkpoints 
    config.run_dir = 'logs/cls_'+str(config.n_way)+'.eps_'+str(config.n_meta_train_episodes) + \
        '.k_shot_' + str(config.k_shot) + '.n_query_' + str(config.n_query)
    # checkpoint_dir = config.experiment_dir + '/' + config.run_dir
    # if checkpoint_name is not None:
    #     config.checkpoint_path = os.path.join(checkpoint_dir, checkpoint_name)
    # else:
    #     config.checkpoint_path = tf.train.latest_checkpoint(checkpoint_dir)

    # create DataLoader for SPLIT
    data_loaders = create_data_loaders(config)

    # load model
    from fillmore.metric_learners import MetricLearner, KNNLearner
    from fillmore.bert_model import BertTextEncoderWrapper, RobertaBinaryClassifier
    config.label_smoothing = 0.0
    config.max_seq_len = 16
    embedding_func = RobertaBinaryClassifier(config)
    model = KNNLearner(embedding_func, distance_type='relevance')
    # embedding_func = BertTextEncoderWrapper(config)
    # model = MetricLearner(embedding_func)
    learning_rate_scheduler = WarmupLRScheduler(config.hidden_size, scale=config.learning_rate, warmup_steps=config.n_meta_train_episodes*config.n_epochs*config.warm_up_prop)
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=learning_rate_scheduler,
        epsilon=config.epsilon,
        clipnorm=config.clipnorm
    )
    checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
    manager = tf.train.CheckpointManager(
        checkpoint, directory=config.run_dir, max_to_keep=3)
    
    # meta train
    if meta_train:
        proto_net_train(config, model, optimizer, retrain=retrain, ckpt_manager=manager)
        # meta validation
        val_stats = proto_net_eval(model, data_loaders["meta_val"], config.n_meta_val_episodes, config)
        if config.oos:
            oos_val_stats = evaluate_oos_f1(model, data_loaders["oos_val"], config.n_meta_val_episodes, config)
            
    # meta test
    if meta_test:
        # load the best checkpoint
        status = checkpoint.restore(manager.latest_checkpoint)
        print("Load model weights from checkpoints {} for meta testing".format(manager.latest_checkpoint))
        stats = proto_net_eval(model, data_loaders["meta_test"], config.n_meta_test_episodes, config)
        print("test acc: {}".format(stats["acc"]))
        print("test loss: {}".format(stats["loss"]))

        if config.oos:
            oos_test_stats = evaluate_oos_f1(model, data_loaders["oos_test"], config.n_meta_val_episodes, config)
            print("oos recall: {}".format(oos_test_stats["oos_recall"]))
            print("in domain acc: {}".format(oos_test_stats["in_domain_accuracy"]))
