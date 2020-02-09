#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path

from prepare_corpora.utils import TextPreprocessor
from utils import read_json_by_item


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--index-path', '-i', type=Path,
                        help='path to an index to be checked')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    preprocessor = TextPreprocessor(filter_empty_brackets=True,
                                    filter_stresses=True,
                                    lowercase=True,
                                    lemmatize=True)

    base_name = args.index_path.name.split('.')[2]
    if '-head-' in base_name:
        base_name = base_name.split('-head-')[0]
    corpus_path = args.index_path.with_name('corpus.' + base_name + '.token.txt')
    if not corpus_path.exists():
        corpus_path = corpus_path.with_suffix('.txt.gz')
    if not corpus_path.exists():
        raise ValueError(f"corpus {corpus_path} doesn't exist.")
    print(f"Loading corpus from {corpus_path}.")
    corpus = [ln.strip() for ln in open(corpus_path, 'rt')]

    print(f"Reading index from {args.index_path} item by item.")
    for lemma, idxs in read_json_by_item(open(args.index_path, 'rt')):
        for idx in idxs:
            if idx[0] > len(corpus):
                print(f"sentence index for lemma '{lemma}' is out of corpus (idx={idx})")
            else:
                sent_tokens = corpus[idx[0]].split()
                if idx[2] > len(sent_tokens):
                    print(f"token index for lemma '{lemma}' is out of corpus (idx={idx})"
                          f", {idx[0]}-th sentence contains {len(sent_tokens)} tokens.")
                else:
                    corpus_token = ' '.join(corpus[idx[0]].split()[idx[1]:idx[2]])
                    corpus_lemma = preprocessor(corpus_token)
                    if corpus_lemma != lemma:
                        print(f"sent {idx[0]} for lemma '{lemma}' contains wrong lemma"
                            f" '{corpus_lemma}' (token='{corpus_token}')")

