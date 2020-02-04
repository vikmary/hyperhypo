#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import re
import itertools
from pathlib import Path
from typing import List, Tuple, Dict, Iterator, Optional
from collections import defaultdict

from hashedindex import textparser
import hashedindex
from tqdm import tqdm

from prepare_corpora.utils import smart_open, count_lines, Lemmatizer, Sanitizer
from utils import get_train_synsets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', '-d', type=Path,
                        help='path to lemmatized corpora')
    parser.add_argument('--train-paths', '-t', type=Path, nargs='+',
                        help='path(s) to text file(s) with training data')
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


if __name__ == "__main__":
    args = parse_args()

    sanitizer = Sanitizer(filter_stresses=True, filter_empty_brackets=True)
    tokenizer = re.compile(r"[\w']+|[^\w ]")
    lemmatizer = Lemmatizer()

    hyponyms = set()
    for synset_id, synset in get_train_synsets(args.train_paths).items():
        for sense in synset['senses']:
            hypo = sanitizer(sense['content'].lower())
            if hypo and (len(tokenizer.findall(hypo)) == 1):
                hyponyms.add(lemmatizer(hypo))
            else:
                print(f"Skipping hyponym {hypo}, because it contains more or less than"
                      f" one token.")
    hyponyms = frozenset(hyponyms)

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

    out_path = args.data_path.with_name('index.full.' + base_name + '.json')
    print(f"Writing full index to {out_path}.")
    json.dump({token: list(idxs) for token, idxs in inverted_index.items()},
              open(out_path, 'wt'),
              indent=2,
              ensure_ascii=False)

    # dumping index for train hyponyms only
    hypo_entries = {h: list(inverted_index.get(h, [])) for h in hyponyms}
    num_entries = len(list(itertools.chain(*hypo_entries.values())))
    print(f"Found {num_entries} hyponym entries,"
          f" {num_entries / len(hypo_entries):.3} per hyponym.")
    n_absent = sum(bool(not idxs) for hypo, idxs in hypo_entries.items())
    print(f"Haven't found context for {n_absent}/{len(hypo_entries)}"
          f" ({int(n_absent/len(hypo_entries) * 100)} %) train hyponyms.")

    out_path = args.data_path.with_name('index.train.' + base_name + '.json')
    print(f"Writing training index to {out_path}.")
    json.dump(hypo_entries, open(out_path, 'wt'), indent=2, ensure_ascii=False)
