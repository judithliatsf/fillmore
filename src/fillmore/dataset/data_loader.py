from fillmore.dataset.loader import load_dataset
import random
import tensorflow as tf

class TextDataLoader(object):

    def __init__(self,
                 data_by_class, 
                 k_shot, 
                 n_query, 
                 n_way,
                 seed=1234,
                 task=None
                 ):
        self.k_shot = k_shot
        self.n_query = n_query
        self.data_by_class = data_by_class
        self.num_support = k_shot * n_way
        self.num_query = n_query * n_way
        self.num_way = n_way
        self.task = task
        self.seed = seed
        random.seed(seed)

    def sample_episodes(self, batch_size):
        """Generate a batch of episodes, where each episode consists of 
        support set and query set

        Args:
            batch_size ([int]): number of episodes

        Returns:
            episodes [List[Dict]]: a list of Episode (a dictionary)
        """
        all_classes = self.data_by_class.keys()
        episodes = []
        for i in range(batch_size):
            episode = {}
            class_names = random.sample(all_classes, self.num_way)
            support_texts, support_labels, support_classes = [], [], []
            query_texts, query_labels, query_classes = [], [], []
            for label_id, class_name in enumerate(class_names):
                # for each class we sample a total of k_shot+n_query examples without replacement
                examples = random.sample(self.data_by_class[class_name], self.k_shot+self.n_query)
                for j, example in enumerate(examples):
                    if j < self.k_shot:
                        support_texts.append(example['raw'])
                        support_labels.append(label_id)
                        support_classes.append(class_name)
                    else:
                        query_texts.append(example['raw'])
                        query_labels.append(label_id)
                        query_classes.append(class_name)
            episode["support_examples"] = support_texts
            episode["query_examples"] = query_texts
            episode["support_labels"] = support_labels
            episode["query_labels"] = query_labels
            episode["support_labels_onehot"] = tf.one_hot(episode["support_labels"], self.num_way)
            episode["query_labels_onehot"] = tf.one_hot(episode["query_labels"], self.num_way)
            episode["support_classes"] = support_classes
            episode["query_classes"] = query_classes
            episode["episode_config"] = {"k_shot": self.k_shot,"n_way": self.num_way,
            "n_query":self.n_query, "task": self.task}
            episodes.append(episode)
        
        return episodes
        

if __name__ == "__main__":
    from transformers import BertConfig
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
    SPLIT="meta_train"
    batch_size = 2
    episodes = data_loaders[SPLIT].sample_episodes(batch_size)
    assert(len(episodes) == 2)
    assert(len(episodes[0]["support_examples"]) == config.k_shot*config.n_way)
    assert(len(episodes[0]["support_labels"]) == config.k_shot*config.n_way)
    assert(episodes[0]["support_labels_onehot"].shape == (config.k_shot*config.n_way, config.n_way))
    assert(len(episodes[0]["query_examples"]) == config.n_query*config.n_way)
    assert(len(episodes[0]["query_labels"]) == config.n_query*config.n_way)
    assert(episodes[0]["query_labels_onehot"].shape == (config.n_query*config.n_way, config.n_way))

