#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
import json
from pathlib import Path
from typing import List, Tuple, Iterator, Optional
from collections import defaultdict

from tqdm import tqdm

from utils import smart_open, count_lines, Lemmatizer, Sanitizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-paths', '-d', type=Path, nargs='+',
                        help='paths to lemmatized corpuses')
    parser.add_argument('--train-path', '-t', type=Path,
                        help='path to text file with training data')
    parser.add_argument('--max-lines', '-l', type=int, default=None,
                        help='take only first l lines of corpus')
    return parser.parse_args()


def find_tokens(tokens: str,
                utterances: Iterator[str],
                progress_bar: tqdm = None,
                max_utterances: Optional[int] = None) -> List[List[Tuple[int, int]]]:
    entries = defaultdict(list)
    for i_u, utter in enumerate(utterances):
        if progress_bar is not None:
            progress_bar.update(1)
        if max_utterances is not None and (i_u >= max_utterances):
            return entries
        for i_t, utter_token in enumerate(utter.split()):
            if utter_token in tokens:
                entries[utter_token].append((i_u, i_t))
    return entries


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
        num_lines = sum(count_lines(fp) for fp in args.data_paths)
    else:
        num_lines = args.max_lines

    entries = {h: [] for h in hyponyms}
    progress_bar = tqdm(total=num_lines, mininterval=10.)
    for fp in args.data_paths:
        with smart_open(fp, 'rt') as fin:
            new_entries = find_tokens(hyponyms, fin,
                                      progress_bar=progress_bar,
                                      max_utterances=num_lines)
        for t, idxs in new_entries.items():
            entries[t].extend(idxs)
        if progress_bar.n >= num_lines:
            break
    progress_bar.close()

    out_path = args.train_path.with_name(args.train_path.name.split('.')[0] +
                                         'index.json')
    print(f"Writing index to {out_path}.")
    json.dump(entries, open(out_path, 'wt'), indent=2, ensure_ascii=False)
                

