#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import itertools
import collections
from pathlib import Path
from typing import List, Tuple, Dict, Iterator, Optional

from tqdm import tqdm

from prepare_corpora.utils import smart_open, count_lines, TextPreprocessor
from utils import get_train_synsets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', '-d', type=Path,
                        help='path to lemmatized corpora')
    parser.add_argument('--train-paths', '-t', type=Path, nargs='+',
                        help='path(s) to text file(s) with training data')
    parser.add_argument('--synset-info-paths', '-s', type=Path, nargs='+',
                        help='paths to synset info for training data')
    parser.add_argument('--max-lines', '-l', type=int, default=None,
                        help='take only first l lines of corpus')
    parser.add_argument('--max-lines-per-item', '-m', type=int, default=300,
                        help='max utterances for each word in index')
    return parser.parse_args()


def build_index(utterances: Iterator[str],
                max_utterances: int,
                max_utterances_per_item: Optional[int] = None) -> Dict[str, collections.Counter]:
    # prioritize_with_max_length: int = 200
    index = collections.defaultdict(collections.Counter)
    max_utters_i = max_utterances_per_item

    for i_u, utter in tqdm(enumerate(utterances), total=max_utterances):
        if i_u >= max_utterances:
            break
        utter_lemmas = utter.split()
        for i_t, utter_lemma in enumerate(utter_lemmas):
            if not max_utters_i or (len(index[utter_lemma]) < max_utters_i):
                index[utter_lemma][(i_u, i_t, i_t+1)] += 1
                # if len(utter_lemmas) < prioritize_with_max_length:
                #     index[utter_lemma][(i_u, i_t)] += 1
            if i_t + 1 < len(utter_lemmas):
                utter_bilemma = ' '.join(utter_lemmas[i_t:i_t+2])
                if not max_utters_i or (len(index[utter_bilemma]) < max_utters_i):
                    index[utter_bilemma][(i_u, i_t, i_t+2)] += 1
                if i_t + 2 < len(utter_lemmas):
                    utter_trilemma = ' '.join(utter_lemmas[i_t:i_t+3])
                    if not max_utters_i or (len(index[utter_trilemma]) < max_utters_i):
                        index[utter_trilemma][(i_u, i_t, i_t+3)] += 1
    return index


if __name__ == "__main__":
    args = parse_args()

    preprocessor = TextPreprocessor(filter_stresses=True,
                                    filter_empty_brackets=True,
                                    lowercase=True,
                                    lemmatize=True)

    hyponyms = set()
    for synset_id, synset in get_train_synsets(args.train_paths, args.synset_info_paths)\
            .items():
        for sense in synset['senses']:
            hypo = preprocessor(sense['content'])
            if not hypo:
                print(f"Skipping hyponym {hypo}, because it's lemma is empty.")
            else:
                hyponyms.add(hypo)
    hyponyms = frozenset(hyponyms)

    if args.max_lines is None:
        print(f"Counting number of input lines.")
        num_lines = count_lines(args.data_path)
    else:
        num_lines = args.max_lines

    with smart_open(args.data_path, 'rt') as fin:
        inverted_index = build_index(fin,
                                     max_utterances=num_lines,
                                     max_utterances_per_item=args.max_lines_per_item)

    # dumping full index
    base_name = args.data_path.name.split('.')[0]
    if args.max_lines is not None:
        base_name += f'-head-{args.max_lines}'

    out_path = args.data_path.with_name('index.full.' + base_name + '.json')
    print(f"Writing full index to {out_path}.")
    json.dump({lemma: list(idxs) for lemma, idxs in inverted_index.items()},
              open(out_path, 'wt'),
              indent=2,
              ensure_ascii=False)

    # dumping index for train hyponyms only
    hypo_entries = {h: list(inverted_index.get(h, [])) for h in hyponyms}
    num_entries = len(list(itertools.chain(*hypo_entries.values())))
    print(f"Found {num_entries} hyponym mentions,"
          f" {int(num_entries / len(hypo_entries))} per hyponym on average.")
    n_absent = sum(bool(not idxs) for hypo, idxs in hypo_entries.items())
    print(f"Haven't found context for {n_absent}/{len(hypo_entries)}"
          f" ({int(n_absent/len(hypo_entries) * 100)}%) train hyponyms.")

    out_path = args.data_path.with_name('index.train.' + base_name + '.json')
    print(f"Writing training index to {out_path}.")
    json.dump(hypo_entries, open(out_path, 'wt'), indent=2, ensure_ascii=False)
