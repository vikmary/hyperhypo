#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from collections import defaultdict

from tqdm import tqdm

from utils import get_train_synsets, get_wordnet_synsets, enrich_with_wordnet_relations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', '-d', type=Path,
                        help='path to a labeled dataset')
    parser.add_argument('--wordnet-dir', '-w', type=Path,
                        help='path to a wordnet directory')
    parser.add_argument('--output_dir', '-o', type=Path,
                        help='output directory for labels prepared for scoring')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    train_synsets = get_train_synsets([args.data_path])

    synsets = get_wordnet_synsets(args.wordnet_dir.glob('synsets.*'))
    enrich_with_wordnet_relations(synsets, args.wordnet_dir.glob('synset_relations.*'))

    data_base_name = args.data_path.name
    out_mix_path = args.output_dir / (data_base_name + '_mix.tsv')
    out_direct_path = args.output_dir / (data_base_name + '_direct.tsv')
    print(f"Writing labels to {out_mix_path}, {out_direct_path}.")

    with open(out_mix_path, 'wt') as f_mix, open(out_direct_path, 'wt') as f_direct:
        for synset_id, synset in tqdm(train_synsets.items()):
            hyperhypernyms = {h['id']: [hh['id']
                                        for hh in synsets[h['id']].get('hypernyms', [])]
                              for h in synset['hypernyms']}
            hypernym_ids = []
            for hh_cand_id in hyperhypernyms:
                for h_id in set(hyperhypernyms) - set([hh_cand_id]):
                    if hh_cand_id in hyperhypernyms[h_id]:
                        break
                else:
                    hypernym_ids.append(h_id)
            for sense_d in synset['senses']:
                sense = sense_d['content']
                for h_id in hypernym_ids:
                    hh_ids_str = '\t'.join(hyperhypernyms[h_id])
                    f_direct.write(f'{sense}\t{h_id}\n')
                    f_mix.write(f'{sense}\t{h_id}\t{hh_ids_str}\n')

