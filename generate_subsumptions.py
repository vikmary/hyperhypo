#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import argparse
from pathlib import Path
from typing import Iterator, Tuple

from utils import get_train_synsets, get_synsets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-data-paths', '-t', type=Path, nargs='+',
                        help='path(s) to files with training data')
    parser.add_argument('--out-path', '-o', type=Path,
                        help='path to a output sumsamptions file')
    parser.add_argument('--ruwordnet-dir', '-w', type=Path,
                        help='path to a ruwordnet directory')
    parser.add_argument('--filter-hyponym-phrases', action='store_true',
                        help='whether to filter hyponyms consisting of'
                        ' multiple words')
    parser.add_argument('--filter-hypernym-phrases', action='store_true',
                        help='whether to filter hypernyms consisting of'
                        ' multiple words')
    return parser.parse_args()


def write_subsumptions(subsumptions: Iterator[Tuple[str, str]], fname: str) -> None:
    with open(fname, 'wt') as f:
        writer = csv.writer(f, dialect='excel-tab', lineterminator='\n')
        for pair in subsumptions:
            writer.writerow(pair)


if __name__ == "__main__":
    args = parse_args()

    train_synsets = get_train_synsets(args.train_data_paths)

    all_synsets = get_synsets(args.ruwordnet_dir.glob('synsets*'))

    subsumptions = ((s['content'].lower(), hyper_s['content'].lower())
                    for synset in train_synsets.values()
                    for s in synset['senses']
                    for hyper_synset in synset['hypernyms']
                    for hyper_s in all_synsets[hyper_synset['id']]['senses'])
    if args.filter_hyponym_phrases:
        subsumptions = filter(lambda h: len(h[0].split()) == 1, subsumptions)
    if args.filter_hypernym_phrases:
        subsumptions = filter(lambda h: len(h[1].split()) == 1, subsumptions)

    write_subsumptions(subsumptions, fname=args.out_path)
