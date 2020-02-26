#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import tqdm
import collections
from pathlib import Path
from typing import Union, Dict, List, Tuple

from utils import get_test_senses
from prepare_corpora.utils import TextPreprocessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern-pairs', type=Path,
                        help='path to a file with hyponym-hypernym pairs'
                        ' extracted from text.')
    parser.add_argument('--data-path', type=Path,
                        help='path to a file with hyponym phrases')
    return parser.parse_args()


def load_extracted_pairs(path: Union[str, Path]) -> Dict[str, List[Tuple[str, int]]]:
    hypo2hypers = collections.defaultdict(list)
    with open(path, 'rt') as fin:
        fin.readline()
        for ln in fin:
            hypo_phr, hyper_phr, freq = ln.split('\t')
            hypo2hypers[hypo_phr].append((hyper_phr, int(freq.strip())))
    hypo2hypers = {k: sorted(values, key=lambda v: v[1], reverse=True)
                   for k, values in hypo2hypers.items()}
    return hypo2hypers


if __name__ == "__main__":
    args = parse_args()

    preprocessor = TextPreprocessor('regexp+pymorphy',
                                    filter_stresses=True,
                                    filter_empty_brackets=True,
                                    lemmatize=True,
                                    lowercase=True)

    hypo2hyper_phrs = load_extracted_pairs(args.pattern_pairs)
    print(f"Preprocessing hypo entries.")
    hypo_norm2hypos = {}
    for hypo in tqdm.tqdm(hypo2hyper_phrs):
        _, hypo_norm = preprocessor(hypo)
        hypo_norm2hypos[hypo_norm] = hypo_norm2hypos.get(hypo_norm, []) + [hypo]

    test_phrs = [sense['content'].lower() for sense in get_test_senses([args.data_path])]

    skipped = 0
    for phr in test_phrs:
        hyper_phrs = []
        if phr in hypo2hyper_phrs:
            hyper_phrs = hypo2hyper_phrs[phr]
        else:
            _, phr_norm = preprocessor(phr)
            if phr_norm in hypo_norm2hypos:
                hyper_phrs = [hyper
                              for hypo in hypo_norm2hypos[phr_norm]
                              for hyper in hypo2hyper_phrs[hypo]]

        if not hyper_phrs:
            skipped += 1

    found = len(test_phrs) - skipped
    print(f"Found hyper candidate phrases for {found}/{len(test_phrs)}"
          f" ({int(found/len(test_phrs) * 100)}%).")
