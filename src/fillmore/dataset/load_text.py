
from fillmore.dataset.loader import load_dataset
import random
import numpy as np

class TextDataGenerator(object):

    def __init__(self, 
                 num_classes, 
                 num_samples_per_class,
                 num_meta_test_classes, 
                 num_meta_test_samples_per_class, 
                 dataset,
                 seed=123
                 ):
        self.num_samples_per_class = num_samples_per_class
        self.num_classes = num_classes
        self.num_meta_test_samples_per_class = num_meta_test_samples_per_class
        self.num_meta_test_classes = num_meta_test_classes
        self.dataset = dataset
        # text_size = config.get('emb_size', 786)

        self.dim_input = 1 # "string"
        self.dim_output = self.num_classes

        random.seed(seed)

    def sample_batch(self, args, batch_type, batch_size):
        """Generate a batch of data

        Args:
            args ([type]): [description]
            batch_type ([str]): [description]
            batch_size ([int]): [description]

        Returns:
            A a tuple of (1) Text batch and (2) Label batch where
            text batch has shape [B, N, K] and label batch has shape [B, N, K, N] if swap is False
            where B is batch size, K is number of samples per class, N is number of classes
        """
        train_classes, val_classes, test_classes, data_by_class = load_dataset(args)
        if batch_type == "meta_train":
            folders = train_classes
            num_classes = self.num_classes
            num_samples_per_class = self.num_samples_per_class
        elif batch_type == "meta_val":
            folders = val_classes
            num_classes = self.num_classes
            num_samples_per_class = self.num_samples_per_class
        else:
            folders = test_classes
            num_classes = self.num_meta_test_classes
            num_samples_per_class = self.num_meta_test_samples_per_class
        
        all_text_batches, all_label_batches = [], []

        for i in range(batch_size):
            sampled_classes = random.sample(folders, num_classes)
            texts, labels = [], []
            for i, c in enumerate(sampled_classes):
                all_items = random.sample(data_by_class[c], num_samples_per_class)
                for item in all_items:
                    texts.append(" ".join(item['text']))
                    labels.append(i)
            texts = np.array(texts).astype(np.str)
            texts = np.reshape(texts, (num_classes, num_samples_per_class,))
            labels = np.array(labels).astype(np.int32)
            labels = np.reshape(labels, (num_classes, num_samples_per_class))
            labels = np.eye(num_classes, dtype=np.float32)[labels]
            
            all_text_batches.append(texts)
            all_label_batches.append(labels)

        all_text_batches = np.stack(all_text_batches)
        all_label_batches = np.stack(all_label_batches)
        return all_text_batches, all_label_batches

# if __name__ == "__main__":
#     args = parse_args()
#     print(args)
#     dg = TextDataGenerator(3, 2, 3, 2, args.dataset)
#     all_text_batches, all_label_batches = dg.sample_batch(args, "meta_val", 3)
    