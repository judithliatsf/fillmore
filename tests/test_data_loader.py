import unittest
from fillmore.dataset.data_loader import TextDataLoader
from fillmore.dataset.loader import load_dataset
from transformers import AutoConfig
import os
from copy import deepcopy

class TextDataGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_way = 3
        cls.k_shot = 2
        cls.meta_batch_size = 5
        config = AutoConfig.from_pretrained("bert-base-uncased")
        config.num_labels = 3
        config.vocab_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures/vocab.txt")
        config.max_seq_len = 7
        config.smlmt = False
        cls.config = config

    def test_load_data(self):
        config = deepcopy(self.config)
        config.dataset = "reuters"
        config.data_path = "data/reuters.json"
        data=load_dataset(config)
        data_generator = TextDataLoader(
            data_by_class=data["meta_train"], 
            k_shot=self.k_shot, 
            n_query=self.k_shot,
            n_way = self.n_way, seed=1234, task=config.dataset)
        episodes = data_generator.sample_episodes(self.meta_batch_size)
        self.assertEqual(len(episodes), self.meta_batch_size)
        self.assertEqual(len(episodes[0]["support_examples"]), 3*2)
        self.assertEqual(len(episodes[0]["support_labels"]), 3*2)
        self.assertEqual(episodes[0]["support_labels_onehot"].shape, [3*2, 3])
        self.assertEqual(config.n_train_class, 15)
        self.assertEqual(config.n_val_class, 5)
        self.assertEqual(config.n_test_class, 11)
    
    def test_load_clinc150(self):
        config = deepcopy(self.config)
        config.dataset = "clinc150"
        config.domains = ['banking']
        config.data_path = "data/clinc150.json"
        data=load_dataset(config)
        data_generator = TextDataLoader(
            data_by_class=data["meta_train"], 
            k_shot=self.k_shot, 
            n_query=self.k_shot,
            n_way = self.n_way, seed=1234, task=config.dataset)
        episodes = data_generator.sample_episodes(self.meta_batch_size)
        self.assertEqual(len(episodes), self.meta_batch_size)
        self.assertEqual(len(episodes[0]["support_examples"]), 3*2)
        self.assertEqual(len(episodes[0]["support_labels"]), 3*2)
        self.assertEqual(episodes[0]["support_labels_onehot"].shape, [3*2, 3])
        self.assertEqual(config.n_train_class, 15)
        self.assertEqual(config.n_val_class, 15)
        self.assertEqual(config.n_test_class, 15)

    def test_load_clinc150a(self):
        config = deepcopy(self.config)
        config.dataset = "clinc150a"
        config.data_path = "data/clinc150.json"
        config.domains = []
        config.num_examples_from_class = 20
        data=load_dataset(config)
        data_generator = TextDataLoader(
            data_by_class=data["meta_train"], 
            k_shot=self.k_shot, 
            n_query=self.k_shot,
            n_way = self.n_way, seed=1234, task=config.dataset)
        episodes = data_generator.sample_episodes(self.meta_batch_size)
        self.assertEqual(len(episodes), self.meta_batch_size)
        self.assertEqual(len(episodes[0]["support_examples"]), 3*2)
        self.assertEqual(len(episodes[0]["support_labels"]), 3*2)
        self.assertEqual(episodes[0]["support_labels_onehot"].shape, [3*2, 3])
        self.assertEqual(config.n_train_class, 100)
        self.assertEqual(config.n_val_class, 20)
        self.assertEqual(config.n_test_class, 30)

    def test_load_smlmt(self):
        config = deepcopy(self.config)
        config.dataset = "clinc150c"
        config.data_path = "data/clinc150.json"
        config.smlmt = True
        config.domains = []
        config.num_examples_from_class_train = 50
        config.num_examples_from_class_valid = 50
        config.num_examples_from_class_test = 50
        data=load_dataset(config)
        data_generator = TextDataLoader(
            data_by_class=data["meta_train"], 
            k_shot=self.k_shot, 
            n_query=self.k_shot,
            n_way = self.n_way, seed=1234, task=config.dataset)
        episodes = data_generator.sample_episodes(self.meta_batch_size)
        self.assertEqual(len(episodes), self.meta_batch_size)
        self.assertEqual(len(episodes[0]["support_examples"]), 3*2)
        self.assertEqual(len(episodes[0]["support_labels"]), 3*2)
        self.assertEqual(episodes[0]["support_labels_onehot"].shape, [3*2, 3])
        self.assertEqual(config.n_smlmt_train_class, 217)
        self.assertEqual(config.n_smlmt_val_class, 0)
        self.assertEqual(config.n_smlmt_test_class, 0)
    
    def test_load_oos(self):
        config = deepcopy(self.config)
        config.dataset = "clinc150b"
        config.domains = ['banking']
        config.data_path = "data/clinc150.json"
        config.oos = True
        config.oos_data_path = "data/clinc150_oos.json"
        data = load_dataset(config)
        data_generator = TextDataLoader(
            data_by_class=data["meta_val"], 
            k_shot=self.k_shot, 
            n_query=self.k_shot,
            n_way = 1,
            oos = config.oos,
            oos_data_by_class=data["oos_val"],
            seed=1234, task=config.dataset)
        episodes = data_generator.sample_episodes(self.meta_batch_size)
        self.assertEqual(len(episodes), self.meta_batch_size)
        self.assertEqual(len(episodes[0]["support_examples"]), 1*2)
        self.assertEqual(len(episodes[0]["support_labels"]), 1*2)
        self.assertEqual(episodes[0]["support_labels_onehot"].shape, [1*2, 1])
        self.assertEqual(len(episodes[0]["oos_examples"]), 2)