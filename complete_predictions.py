#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Union, List, Tuple, Dict

from utils import get_test_senses


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', type=Path,
                        help='path to a file with hyponym phrases')
    parser.add_argument('--pred-path', type=Path,
                        help='path to a file with incomplete predictions')
    return parser.parse_args()


def get_prediction(fname: Union[str, Path]) -> Dict[str, List[Tuple[str, str]]]:
    preds = {}
    with open(fname, 'rt') as fin:
        for ln in fin:
            hyponym, hypernym, other = ln.split('\t', 2)
            other = other.strip()
            if hyponym in preds:
                preds[hyponym].append((hypernym, other))
            else:
                preds[hyponym] = [(hypernym, other)]
    return preds


if __name__ == "__main__":
    args = parse_args()

    test_phrs = [sense['content'] for sense in get_test_senses([args.data_path])]
    preds = get_prediction(args.pred_path)

    base_name, suffix = args.pred_path.name.split('.', 1)
    out_path = args.pred_path.with_name(base_name + '-completed-to-random.' + suffix)
    print(f"Writing completed predictions to {out_path}.")
    skipped = 0
    with open(out_path, 'wt') as fout:
        for test_phr in test_phrs:
            if test_phr not in preds:
                h_id, other = '100022-N', 'РЕЧУШКА,РЕКА,РЕЧКА'
                fout.write(f'{test_phr}\t{h_id}\t{other}\n')
                skipped += 1
            else:
                phr_preds = set()
                for h_id, other in preds[test_phr]:
                    if h_id not in phr_preds:
                        phr_preds.add(h_id)
                        fout.write(f'{test_phr}\t{h_id}\t{other}\n')

    print(f"Found {len(test_phrs) - skipped} out of {len(preds)} predictions.")


