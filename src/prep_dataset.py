"""
Run this script to get prepared dataset
"""

import json
import os
import requests 
import zipfile
import shutil 
import pandas as pd

from transformers import BertTokenizer


def download_url(url, save_path, chunk_size=128):
    r = requests.get(url, stream=True)
    with open(save_path, 'wb') as fd:
        for chunk in r.iter_content(chunk_size=chunk_size):
            fd.write(chunk)

def unzip_file(file_path, extract_path):
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        zip_ref.extractall(extract_path)

# Download and unzip Distributional-Signatures Dataset
def get_distsig_dataset():
    print("downloading from distributional-signatures")
    download_url("https://people.csail.mit.edu/yujia/files/distributional-signatures/data.zip", "../data.zip")
    unzip_file("../data.zip", "..")
    os.remove("../data.zip")
    print("Got data folder")

def get_clinc150_dataset():
    print("downloading from uci")
    download_url("https://archive.ics.uci.edu/ml/machine-learning-databases/00570/clinc150_uci.zip", "../clinc150_uci.zip")
    unzip_file("../clinc150_uci.zip", "..")
    os.remove("../clinc150_uci.zip")

    # reformat into distsig_dataset data folder with BERT tokenizer
    with open('../clinc150_uci/data_full.json') as f:
        data = json.load(f)

    train_dat = pd.DataFrame(data["train"])
    train_dat = train_dat.rename(columns={0:"sample", 1:"label"})
    val_dat = pd.DataFrame(data["val"])
    val_dat = val_dat.rename(columns={0:"sample", 1:"label"})
    test_dat = pd.DataFrame(data["test"])
    test_dat = test_dat.rename(columns={0:"sample", 1:"label"})

    label_map = {}
    for i, label in enumerate(train_dat["label"].unique()):
        label_map[label] = i

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    json_strs = []
    for i in range(len(train_dat)):
        json_dict = {'text': tokenizer.tokenize(train_dat["sample"][i]),
                    'raw': train_dat["sample"][i],
                    'label': label_map[train_dat["label"][i]]}
        json_str = json.dumps(json_dict)
        json_strs.append(json_str+'\n')

    for i in range(len(val_dat)):
        json_dict = {'text': tokenizer.tokenize(val_dat["sample"][i]),
                    'raw': val_dat["sample"][i],
                    'label': label_map[val_dat["label"][i]]}
        json_str = json.dumps(json_dict)
        json_strs.append(json_str+'\n')
        
    for i in range(len(test_dat)):
        json_dict = {'text': tokenizer.tokenize(test_dat["sample"][i]),
                    'raw': test_dat["sample"][i],
                    'label': label_map[test_dat["label"][i]]}
        json_str = json.dumps(json_dict)
        json_strs.append(json_str+'\n')
        
    txtfile = open("../data/clinc150.json", 'w')
    txtfile.writelines(json_strs)
    txtfile.close()
    shutil.rmtree("../clinc150_uci")
    shutil.rmtree("../__MACOSX")

def main():
    get_distsig_dataset()
    get_clinc150_dataset()

if __name__ == "__main__":
    # execute only if run as a script
    main()