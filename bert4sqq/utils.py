# coding=utf-8

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import codecs
import pickle
import re
import os


def load_data(data_dir, file_name):
    lines = []
    with open(os.path.join(data_dir, file_name), encoding='utf-8') as f:
        for (i, line) in enumerate(f):
            lines.append(line)
    return lines


def get_labels(data_dir, output_dir, file_name):
    label_set = set()
    zh_pattern = re.compile(u'[\u4e00-\u9fa5]+')
    with open(os.path.join(data_dir, file_name), encoding='utf-8') as f:
        for (i, line) in enumerate(f):
            splits = line.strip().split("\t")
            label = splits[-1].strip()
            label_set.add(label)

    labels = list(label_set)

    for _, val in enumerate(labels[:]):
        if zh_pattern.search(val):
            # 如果里面有中文
            # tf.logging.info(val)
            labels.remove(val)
        elif val == '':
            labels.remove(val)

    label_map = {}
    for (i, label) in enumerate(labels):
        label_map[label] = i
    if not os.path.exists(os.path.join(output_dir, "label2id.pkl")):
        with open(os.path.join(output_dir, "label2id.pkl"), 'wb') as w:
            pickle.dump(label_map, w)

    return labels


def get_labels2id(label_path):
    with codecs.open(os.path.join(label_path, 'label2id.pkl'), 'rb') as rf:
        label2id = pickle.load(rf)
        return label2id
