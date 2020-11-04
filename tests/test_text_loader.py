import unittest
from fillmore.dataset.load_text import TextDataGenerator
from transformers import AutoConfig
import argparse

# def parse_args():
#     parser = argparse.ArgumentParser(
#             description="Few Shot Text Classification with BERT")

#     # data configuration
#     parser.add_argument("--data_path", type=str,
#                         default="data/reuters.json",
#                         help="path to dataset")
#     parser.add_argument("--dataset", type=str, default="reuters",
#                     help="name of the dataset. "
#                     "Options: [20newsgroup, amazon, huffpost, "
#                     "reuters, rcv1, fewrel]")
#     parser.add_argument("--n_train_class", type=int, default=15,
#                         help="number of meta-train classes")
#     parser.add_argument("--n_val_class", type=int, default=5,
#                         help="number of meta-val classes")
#     parser.add_argument("--n_test_class", type=int, default=11,
#                         help="number of meta-test classes")
#     parser.add_argument("--mode", type=str, default="test",
#                     help=("Running mode."
#                             "Options: [train, test, finetune]"
#                             "[Default: test]"))
#     return parser.parse_args()

class TextDataGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_way = 3
        cls.k_shot = 2
        cls.dataset = "reuters"
        cls.meta_batch_size = 5
        config = AutoConfig.from_pretrained("bert-base-uncased")
        config.mode="finetune"
        config.dataset = "reuters"
        config.n_train_class = 15
        config.n_val_class = 5
        config.n_test_class = 11
        config.data_path = "data/reuters.json"
        cls.config = config

    def test_load_data(self):
        data_generator = TextDataGenerator(self.n_way, self.k_shot*2, self.n_way, self.k_shot*2, self.config)
        texts, labels = data_generator.sample_batch(self.config, "meta_train", self.meta_batch_size)
        self.assertEqual(texts.shape, (5, 3, 4))
        self.assertEqual(labels.shape, (5, 3, 4, 3))