#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path

from utils import get_train_synsets, get_wordnet_synsets, enrich_with_wordnet_relations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-paths', '-t', type=Path, nargs='+',
                        help='path(s) to training data')
    parser.add_argument('--wordnet-dir', '-w', type=Path,
                        help='path to a wordnet directory')
    parser.add_argument('--output_path', '-o', type=Path,
                        help='output file with candidates')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_synsets = get_train_synsets(args.data_paths)

    synsets = get_wordnet_synsets(args.wordnet_dir.glob('synsets.*'))
    synsets["2116-N"]
    enrich_with_wordnet_relations(synsets, args.wordnet_dir.glob('synset_relations.*'))

    train_tuples = []
    for synset_id, synset in train_synsets.items():
        senses = [s['content'].lower() for s in synset['senses']]
        # construct hypernyms
        hypernym_ids = [h['id'] for h in synset['hypernyms']]
        hypernyms = [[s['content'].lower() for s in synsets[h_id]['senses']]
                     for h_id in hypernym_ids]
        # construct hypernyms of hypernyms
        hyperhypernym_ids = set()
        for h_id in hypernym_ids:
            hyperhypernym_ids.update(hh['id']
                                     for hh in synsets[h_id].get('hypernyms', []))
        hyperhypernyms = [[s['content'].lower() for s in synsets[hh_id]['senses']]
                          for hh_id in hyperhypernym_ids]

        train_tuples.append((senses, hypernyms, hyperhypernyms))

    print(f"Writing output json to {args.output_path}.")
    json.dump(train_tuples, open(args.output_path, 'wt'), indent=2, ensure_ascii=False)
