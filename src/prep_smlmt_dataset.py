"""
Run this script to get SMLMT dataset for dataset in data folder
"""

import json
import os
import glob
import pandas as pd
from collections import defaultdict


def get_smlmt(data_path):
    print("reading "+data_path)

    word_dict = defaultdict(int) 
    linc = 0
    all_sentences = set()
    # for each word, how man sentences contain this word
    with open(data_path, 'r', errors='ignore') as f:
        data = []
        data_by_class = {}
        for line in f:
            linc += 1
            linset = {}
            row = json.loads(line)
            all_sentences.add(" ".join(row['text']))
            for word in row['text']:
                if word not in linset:
                    word_dict[word] += 1

    # how many words (v) appeared in how many sentences (k)
    wcd = defaultdict(int) 
    for c in word_dict.values():
        wcd[c] += 1
    
    # take words with more than 30 samples smaller than 100 samples
    n_samples = 30
    label = 0
    label_map = {}
    json_strs = []
    labeled_sentences = set()
    while n_samples < 100:
        for w in [w for w,c in word_dict.items() if c == n_samples and len(w)>=3]: 
            label_map[w] = label
            for sentence in all_sentences:
                if sentence not in labeled_sentences:
                    word_list = sentence.split()
                    # filter out masking word
                    token_list = list(filter(lambda token: token != w, word_list))
                    json_dict = {'text': token_list,
                                'label': label}
                    json_str = json.dumps(json_dict)
                    json_strs.append(json_str+'\n')
                    labeled_sentences.add(sentence)
            label += 1
        n_samples += 1

    print("writing "+"smlmt_"+data_path)
    txtfile = open("smlmt_"+data_path, 'w')
    txtfile.writelines(json_strs)
    txtfile.close()
    
    label_str = json.dumps(label_map)
    labelfile = open("smlmt_label_"+data_path, 'w')
    labelfile.writelines(json_strs)
    labelfile.close()

def main():
    os.chdir("../data")
    data_files = glob.glob("*.json")
    for f in data_files:
        get_smlmt(f)


if __name__ == "__main__":
    # execute only if run as a script
    main()