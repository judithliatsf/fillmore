"""
Run this script to get SMLMT dataset for dataset in data folder
"""

import json
import os
import pandas as pd
from collections import defaultdict
from fillmore.dataset.utils import tprint

from spacy.tokenizer import Tokenizer
from spacy.lang.en import English
nlp = English()
tokenizer = Tokenizer(nlp.vocab)

def get_smlmt(examples):

    all_sentences = set()
    word_dict = {}

    for row in examples:
        all_sentences.add(row["raw"])
        doc = tokenizer(row["raw"])
        tokens = [token.text for token in doc]
        for token in doc:
            word = token.text
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
    data_by_class = {}
    json_strs = []
    labeled_sentences = set()
    no_label_map = {}
    for w, items in word_dict.items():

        if len(w) >=3 and len(items) >=30 and len(items) < 100:
            has_example = False
            for item in items:
                if item["raw"] not in labeled_sentences:
                    item["label"] = label
                    # add items under the same class label
                    if label in data_by_class.keys():
                        data_by_class[label].append(item)
                    else:
                        data_by_class[label] = [item]
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
    return data_by_class, json_strs, label_map, no_label_map

def build_smlmt(examples, config):
    """Create smlmt tasks from unlabeled examples
    """
    
    folder = os.path.dirname(config.data_path)
    task_name = config.dataset

    # save smlmt examples for target task
    json_strs = [json.dumps(item)+'\n' for item in examples]
    smlmt_data_path = os.path.join(folder, task_name + "_pre_smlmt.json")
    txtfile = open(smlmt_data_path, 'w')
    txtfile.writelines(json_strs)
    txtfile.close()
    tprint('saving {} unlabeled examples to file: {}'.format(len(examples), smlmt_data_path))

    # create smlmt tasks from unlabeled examples
    data_by_class, json_strs, label_map, no_label_map = get_smlmt(examples)

    tprint("writing smlmt tasks to "+"smlmt_" + task_name + ".json")
    txtfile = open(os.path.join(folder, "smlmt_" + task_name + ".json"), 'w')
    txtfile.writelines(json_strs)
    txtfile.close()
    
    tprint("writing words included in smlmt tasks to "+"smlmt_label_" + task_name + ".json")
    label_str = json.dumps(label_map)
    labelfile = open(os.path.join(folder, "smlmt_label_" + task_name + ".json"), 'w')
    labelfile.writelines(label_str)
    labelfile.close()

    tprint("writing words not included in smlmt tasks to "+"smlmt_no_label_" + task_name + ".json")
    no_label_str = json.dumps(no_label_map)
    labelfile = open(os.path.join(folder, "smlmt_no_label_" + task_name + ".json"), 'w')
    labelfile.writelines(no_label_str)
    labelfile.close()
    return data_by_class

