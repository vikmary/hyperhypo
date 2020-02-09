#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import csv
from typing import Union, Dict
from pathlib import Path

from utils import get_train_synsets, get_wordnet_synsets, enrich_with_wordnet_relations
from utils import get_all_related


def write_training(synsets: Dict[str, Dict],
                   fname: Union[str, Path]) -> None:
    with open(fname, 'wt') as fout:
        writer = csv.writer(fout, delimiter='\t')
        writer.writerow(['SYNSET_ID', 'TEXT', 'PARENT', 'PARENT_TEXTS'])
        for synset_id, synset in synsets.items():
            text = ','.join(s['content'] for s in synset['senses'])
            parents = "['" + "', '".join(h['id'] for h in synset['hypernyms']) + "']"
            writer.writerow([synset_id, text, parents, ''])
            if 'hyperhypernyms' in synset:
                parents = "['" + "', '".join(h['id']
                                             for h in synset['hyperhypernyms']) + "']"
                writer.writerow([synset_id, text, parents, ''])


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', '-t', type=Path,
                        help='path to training data')
    parser.add_argument('--wordnet-dir', '-w', type=Path,
                        help='path to a wordnet directory')
    parser.add_argument('--seed', '-s', type=int, default=131,
                        help='random seed initial value')
    parser.add_argument('--valid-rate', '-r', type=float, default=0.2,
                        help='validation/full ratio')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)

    train_synsets = get_train_synsets([args.data_path])

    synsets = get_wordnet_synsets(args.wordnet_dir.glob('synsets.*'))
    enrich_with_wordnet_relations(synsets, args.wordnet_dir.glob('synset_relations.*'))

    valid_synsets = {}
    num_old_train = len(train_synsets)
    i = 0
    while len(valid_synsets) < args.valid_rate * num_old_train:
        i += 1
        synset_id = random.choice(list(train_synsets.keys()))
        synset = train_synsets.pop(synset_id)
        valid_synsets[synset_id] = synset

        directly_related = [h['id'] for h in synset['hypernyms']]
        for dr_synset_id in directly_related:
            for r_synset_id in get_all_related(dr_synset_id, synsets,
                                               relation_types=['POS-synonym',
                                                               'hypernyms']):
                if r_synset_id in train_synsets:
                    valid_synsets[r_synset_id] = train_synsets.pop(r_synset_id)

    print(f"Generated {len(valid_synsets)} / {num_old_train}"
          f"({len(valid_synsets)/num_old_train:.4}) validation synsets"
          f" from {i} random synsets.")

    train_out_path = args.data_path.with_name(args.data_path.name.rsplit('.', 1)[0] +
                                              '.train.tsv')
    print(f"Writing training synsets to {train_out_path}.")
    write_training(train_synsets, train_out_path)

    valid_out_path = args.data_path.with_name(args.data_path.name.rsplit('.', 1)[0] +
                                              '.valid.tsv')
    print(f"Writing validation synsets to {valid_out_path}.")
    write_training(valid_synsets, valid_out_path)
