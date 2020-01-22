#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import itertools
from pathlib import Path
from typing import List, Tuple, Dict, Iterator, Optional
from collections import defaultdict

from hashedindex import textparser
import hashedindex
from tqdm import tqdm

from utils import smart_open, count_lines, Lemmatizer, Sanitizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', '-d', type=Path,
                        help='path to lemmatized corpora')
    parser.add_argument('--train-path', '-t', type=Path,
                        help='path to text file with training data')
    parser.add_argument('--max-lines', '-l', type=int, default=None,
                        help='take only first l lines of corpus')
    return parser.parse_args()


def build_index(utterances: Iterator[str],
                max_utterances: int) -> Dict[str, List[Tuple[int, int]]]:
    index = hashedindex.HashedIndex()

    for i_u, utter in tqdm(enumerate(utterances), total=max_utterances):
        if i_u >= max_utterances:
            break
        for i_t, utter_token in enumerate(utter.split()):
            index.add_term_occurrence(utter_token, (i_u, i_t))
    return index.items()


def read_training_data(fp: Path) -> List[Tuple[str, str]]:
    reader = csv.reader(open(fp, 'rt'), delimiter='\t')
    data = []
    for row in reader:
        if len(row) != 2:
            raise ValueError(f'input file in the wrong format. Row {row} is invalid.')
        data.append(row)
    return data


if __name__ == "__main__":
    args = parse_args()

    sanitizer = Sanitizer(filter_diacritical=True, filter_empty_brackets=True)
    lemmatizer = Lemmatizer()

    train_data = read_training_data(args.train_path)
    hyponyms = []
    for hypo, hyper in train_data:
        hypo = lemmatizer(sanitizer(hypo))
        if hypo and (len(hypo.split()) == 1):
            hyponyms.append(hypo)
        else:
            print(f"Skipping hyponym {hypo}, because it contains more or less than"
                  f" one token.")

    if args.max_lines is None:
        print(f"Counting number of input lines.")
        num_lines = count_lines(args.data_path)
    else:
        num_lines = args.max_lines

    with smart_open(args.data_path, 'rt') as fin:
        inverted_index = build_index(fin, max_utterances=num_lines)

    # dumping full index
    base_name = args.data_path.name.split('.')[0]
    if args.max_lines is not None:
        base_name += f'-head-{args.max_lines}'

    out_path = args.data_path.with_name(base_name + '.index-full.json')
    print(f"Writing full index to {out_path}.")
    json.dump({token: list(idxs) for token, idxs in inverted_index.items()},
              open(out_path, 'wt'), indent=2, ensure_ascii=False)

    # dumping index for train hyponyms only
    hypo_entries = {h: list(inverted_index.get(h, [])) for h in hyponyms}
    num_entries = len(list(itertools.chain(*hypo_entries.values())))
    print(f"Found {num_entries} hyponym entries,"
          f" {num_entries / len(hypo_entries):.3} per hyponym.")

    out_path = args.data_path.with_name(base_name + '.index-train.json')
    print(f"Writing training index to {out_path}.")
    json.dump(hypo_entries, open(out_path, 'wt'), indent=2, ensure_ascii=False)
