"""
Run this script to get SMLMT dataset for dataset in data folder
"""

import json
import os
import glob
import pandas as pd
from collections import defaultdict

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')

def get_smlmt(data_path, filename):
    data_path = os.path.join(data_path, filename)
    print("reading "+data_path)

    linc = 0
    all_sentences = set()
    # for each word, how man sentences contain this word
    with open(data_path, 'r', errors='ignore') as f:
        data = []
        word_dict = {} # dict with token as key, and list of sentences as values
        for line in f:
            linc += 1
            row = json.loads(line)
            all_sentences.add(row["raw"])
            tokens = tokenizer.tokenize(row["raw"])
            for word in tokens:
                json_dict = {
                    "text": tokens,
                    "raw": row["raw"].replace(word, "[MASK]"),
                    "word": word 
                }
                if word not in word_dict:
                    word_dict[word] = [json_dict]
                else:
                    word_dict[word].append(json_dict)
    
    # take words with more than 30 samples smaller than 100 samples
    label = 0
    label_map = {}
    json_strs = []
    labeled_sentences = set()
    no_label_map = {}
    for w, items in word_dict.items():

        if len(w) >=3 and len(items) >=30 and len(items) < 100:
            has_example = False
            for item in items:
                if item["raw"] not in labeled_sentences:
                    item["label"] = label
                    json_str = json.dumps(item)
                    json_strs.append(json_str+'\n')
                    labeled_sentences.add(item["raw"])
                    has_example = True
            if has_example:
                label_map[w] = label            
                label = label + 1
            else:
                no_label_map[w] = len(items)
        else:
            no_label_map[w] = len(items)

    print("writing "+"smlmt_"+filename)
    txtfile = open("smlmt_"+filename, 'w')
    txtfile.writelines(json_strs)
    txtfile.close()
    
    label_str = json.dumps(label_map)
    labelfile = open("smlmt_label_"+filename, 'w')
    labelfile.writelines(label_str)
    labelfile.close()

    no_label_str = json.dumps(no_label_map)
    labelfile = open("smlmt_no_label_"+filename, 'w')
    labelfile.writelines(no_label_str)
    labelfile.close()

def main():
    os.chdir("../data")
    data_files = glob.glob("*.json")
    for f in data_files:
        get_smlmt("../data", f)


if __name__ == "__main__":
    # execute only if run as a script
    # main()

    # execute from root
    get_smlmt("data/", "clinc150small.json")