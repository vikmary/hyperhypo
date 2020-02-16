#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import collections
from pathlib import Path

from corpus_indexed import CorpusIndexed
from prepare_corpora.utils import TextPreprocessor, count_lines, smart_open


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

    corpus_path = CorpusIndexed.get_corpus_path(args.index_path, level='token')
    print(f"Counting lines in {corpus_path}.")
    num_lines = count_lines(corpus_path)

    print(f"Loading index from {args.index_path}.")
    index, sent_idxs = CorpusIndexed.load_index(args.index_path)
    sent2idxs = collections.defaultdict(list)
    for lemma, idxs in index.items():
        for idx in idxs:
            if idx[0] > num_lines:
                print(f"sentence index for lemma '{lemma}' is out of corpus (idx={idx})")
            sent2idxs[idx[0]].append((lemma, idx[1], idx[2]))
    del index

    print(f"Loading corpus from {corpus_path} line by line.")
    with smart_open(corpus_path, 'rt') as fin:
        for sent_idx, ln in enumerate(fin):
            if sent_idx not in sent_idxs:
                continue
            for lemma, h_start, h_end in sent2idxs[sent_idx]:
                sent_tokens = ln.split()
                if h_end > len(sent_tokens):
                    print(f"token index for lemma '{lemma}' is out of corpus"
                          f" (idx[-1]={h_end}), "
                          f"{sent_idx}-th sentence contains {len(sent_tokens)} tokens.")
                else:
                    corpus_token = ' '.join(sent_tokens[h_start:h_end])
                    corpus_lemma = preprocessor(corpus_token)
                    if corpus_lemma != lemma:
                        print(f"sent {sent_idx} for lemma '{lemma}' contains wrong lemma"
                            f" '{corpus_lemma}' (token='{corpus_token}')")

