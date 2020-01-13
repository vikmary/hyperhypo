#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys

from utils import get_train_synsets, get_test_senses, get_hyperstar_senses
from utils import synsets2senses


if __name__ == "__main__":
    data_path, hyperstar_data_path = sys.argv[1:]
    if 'train' in data_path:
        synsets = get_train_synsets([data_path])
        senses = synsets2senses(synsets)
    else:
        senses = get_test_senses([data_path])
    words = set(s['content'].lower() for s in senses)

    hyperstar_senses = get_hyperstar_senses([hyperstar_data_path])
    hyperstar_words = set(s['content'].lower() for s in hyperstar_senses)

    sys.stderr.write(f"{len(words):10} words in {data_path}.\n")
    sys.stderr.write(f"{len(hyperstar_words):10} words in {hyperstar_data_path}.\n")
    sys.stderr.write(f"{len(words & hyperstar_words):10} words in intersection.\n")

    
    
