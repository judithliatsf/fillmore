import datetime
import pandas as pd
from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
tokenizer = Tokenizer(nlp.vocab)

def tprint(s):
    '''
        print datetime and s
        @params:
            s (str): the string to be printed
    '''
    print('{}: {}'.format(
        datetime.datetime.now().strftime('%02y/%02m/%02d %H:%M:%S'), s),
          flush=True)


def data_from_df(df):
    data_by_class = {}
    oos_data_by_class = {}
    classes = df.columns
    for c in classes:
        examples = []
        
        for example in df[c].to_list():
            intent_example = {} 
            if isinstance(example, str):
                intent_example["label"] = c
                intent_example["raw"] = example.lower()
                intent_example["text"] = [token for token in tokenizer(intent_example["raw"])]
                examples.append(intent_example)
    
        if c == "OOS":
            oos_data_by_class[c] = examples
        else:
            data_by_class[c] = examples

    return data_by_class, oos_data_by_class

if __name__ == "__main__":
    with open ("data/salesforce/sales.tsv", encoding="utf-8", mode="r") as f:
        df = pd.read_csv(f, sep="\t")
    data_by_class, oos_data_by_class = data_from_df(df)

    from transformers import RobertaConfig
    config=RobertaConfig.from_dict({
    "dataset": "sales",
    "data_path": "/dbfs/judith/fillmore/clinc150.json",
    "num_examples_from_class_train": 50,
    "num_examples_from_class_valid": 50,
    "num_examples_from_class_test": 50,
    "n_way": 9, # 6 for non-smlmt, 9 for smlmt
    "k_shot": 10,
    "n_query": 10,
    "n_meta_test_way": 4,
    "k_meta_test_shot": 10,
    "n_meta_test_query": 10,
    "oos": True,
    "oos_data_path": "/dbfs/judith/fillmore/clinc150_oos.json", 
    "smlmt": True,
    "smlmt_ratio": 0.6,
    "smlmt_k_shot": 15,
    "smlmt_n_query": 10,
    "seed": 1234
})
    from fillmore.dataset.data_loader import TextDataLoader
    data_loader = TextDataLoader(
    data_by_class,
    config.k_meta_test_shot, config.n_meta_test_query, config.n_meta_test_way,
    seed=config.seed, task=config.dataset,
    oos=config.oos, oos_data_by_class=oos_data_by_class
)
    samples = data_loader.sample_episodes(1)[0]
