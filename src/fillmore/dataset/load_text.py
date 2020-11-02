
from fillmore import dataset
from fillmore.dataset.loader import load_dataset
import argparse
import random

def parse_args():
    parser = argparse.ArgumentParser(
            description="Few Shot Text Classification with BERT")

    # data configuration
    parser.add_argument("--data_path", type=str,
                        default="data/reuters.json",
                        help="path to dataset")
    parser.add_argument("--dataset", type=str, default="reuters",
                    help="name of the dataset. "
                    "Options: [20newsgroup, amazon, huffpost, "
                    "reuters, rcv1, fewrel]")
    parser.add_argument("--n_train_class", type=int, default=15,
                        help="number of meta-train classes")
    parser.add_argument("--n_val_class", type=int, default=5,
                        help="number of meta-val classes")
    parser.add_argument("--n_test_class", type=int, default=11,
                        help="number of meta-test classes")
    parser.add_argument("--mode", type=str, default="test",
                    help=("Running mode."
                            "Options: [train, test, finetune]"
                            "[Default: test]"))
    return parser.parse_args()

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
            for c in sampled_classes:
                all_items = random.sample(data_by_class[c], num_samples_per_class)
                for item in all_items:
                    texts.append(" ".join(item['text']))
                    labels.append(item['label'])

            all_text_batches.append(texts)
            all_label_batches.append(labels)
        
        return all_text_batches, all_label_batches

if __name__ == "__main__":
    args = parse_args()
    print(args)
    dg = TextDataGenerator(3, 1, 3, 1, args.dataset)
    all_text_batches, all_label_batches = dg.sample_batch(args, "meta_val", 3)
    import pdb; pdb.set_trace()
    