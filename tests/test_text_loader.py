import unittest
from fillmore.dataset.load_text import TextDataGenerator
from transformers import AutoConfig
import os

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
        cls.config = config

    def test_load_data(self):
        self.config.dataset = "reuters"
        self.config.data_path = "data/reuters.json"
        data_generator = TextDataGenerator(self.n_way, self.k_shot*2, self.n_way, self.k_shot*2, self.config)
        texts, labels = data_generator.sample_batch(self.config, "meta_train", self.meta_batch_size)
        self.assertEqual(texts.shape, (5, 3, 4))
        self.assertEqual(labels.shape, (5, 3, 4, 3))
        self.assertEqual(self.config.n_train_class, 15)
        self.assertEqual(self.config.n_val_class, 5)
        self.assertEqual(self.config.n_test_class, 11)
    
    def test_load_clinc150(self):
        self.config.dataset = "clinc150"
        self.config.domains = ['banking']
        self.config.data_path = "data/clinc150.json"
        data_generator = TextDataGenerator(self.n_way, self.k_shot*2, self.n_way, self.k_shot*2, self.config)
        texts, labels = data_generator.sample_batch(self.config, "meta_train", self.meta_batch_size)
        self.assertEqual(texts.shape, (5, 3, 4))
        self.assertEqual(labels.shape, (5, 3, 4, 3))
        self.assertEqual(self.config.n_train_class, 15)
        self.assertEqual(self.config.n_val_class, 15)
        self.assertEqual(self.config.n_test_class, 15)

    def test_load_clinc150a(self):
        self.config.dataset = "clinc150a"
        self.config.data_path = "data/clinc150.json"
        self.config.domains = []
        self.config.num_examples_from_class = 20
        data_generator = TextDataGenerator(self.n_way, self.k_shot*2, self.n_way, self.k_shot*2, self.config)
        texts, labels = data_generator.sample_batch(self.config, "meta_train", self.meta_batch_size)
        self.assertEqual(texts.shape, (5, 3, 4))
        self.assertEqual(labels.shape, (5, 3, 4, 3))
        self.assertEqual(self.config.n_train_class, 100)
        self.assertEqual(self.config.n_val_class, 20)
        self.assertEqual(self.config.n_test_class, 30)

    def test_load_smlmt(self):
        self.config.dataset = "smlmt"
        self.config.data_path = "data/smlmt_clinc150small.json"
        self.config.domains = []
        self.config.num_examples_from_class = 20
        data_generator = TextDataGenerator(self.n_way, self.k_shot*2, self.n_way, self.k_shot*2, self.config)
        texts, labels = data_generator.sample_batch(self.config, "meta_train", self.meta_batch_size)
        self.assertEqual(texts.shape, (5, 3, 4))
        self.assertEqual(labels.shape, (5, 3, 4, 3))
        self.assertEqual(self.config.n_train_class, 325)
        self.assertEqual(self.config.n_val_class, 0)
        self.assertEqual(self.config.n_test_class, 0)

    def test_encode_text(self):
        self.config.dataset = "reuters"
        self.config.data_path = "data/reuters.json"

        from fillmore.bert_model import BertTextEncoderWrapper
        encoder = BertTextEncoderWrapper(self.config)
        data_generator = TextDataGenerator(self.n_way, self.k_shot*2, self.n_way, self.k_shot*2, self.config, encoder=encoder)
        texts, labels = data_generator.sample_batch(self.config, "meta_train", self.meta_batch_size)
        self.assertEqual(texts.shape, (5, 3, 4, 768))
        self.assertEqual(labels.shape, (5, 3, 4, 3))