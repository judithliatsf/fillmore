import unittest
from fillmore.load_data import DataGenerator

class DataGeneratorTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.n_way = 3
        cls.k_shot = 2
        cls.data_path = './omniglot_resized'
        cls.meta_batch_size = 5

    def test_load_data(self):
        data_generator = DataGenerator(self.n_way, self.k_shot*2, self.n_way, self.k_shot*2, config={'data_folder': self.data_path})
        images, labels = data_generator.sample_batch("meta_train", self.meta_batch_size, shuffle=False, swap=False)
        self.assertEquals(images.shape, (5, 3, 4, 784))
        self.assertEquals(labels.shape, (5, 3, 4, 3))