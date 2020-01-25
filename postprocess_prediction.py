#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
from pathlib import Path
from typing import Union, Tuple, List

from utils import get_wordnet_synsets, enrich_with_wordnet_relations


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prediction-path', '-p', type=Path,
                        help='path to a labeled dataset')
    parser.add_argument('--wordnet-dir', '-w', type=Path,
                        help='path to a wordnet directory')
    return parser.parse_args()


def get_prediction(fname: Union[str, Path]) -> List[Tuple[str, List[str]]]:
    pred_tuples = []
    with open(fname, 'rt') as fin:
        for ln in fin:
            hyponym, hypernym, other = ln.split('\t')
            other = other.strip()
            if pred_tuples and (pred_tuples[-1][0] == hyponym):
                pred_tuples[-1][1].append(hypernym)
            else:
                pred_tuples.append([hyponym, [hypernym]])
    return pred_tuples


def rerank_predictions(hypernyms: List[str], synsets: Dict[str, Dict]) -> List[str]:
    num_old = len(hypernyms)
    hyperhypernyms_d = {h_id: [hh['id']
                               for hh in synsets[h_id].get('hypernyms', [])]
                        for h_id in hypernyms}
    new_hypernyms_head, new_hypernyms_tail = [], []
    while hypernyms:
        h_id = hypernyms[0]
        for hh_id in hyperhypernyms_d[h_id]:
            if hh_id in hypernyms[1:]:
                new_hypernyms_head.append(hh_id)
                hypernyms.remove(hh_id)
                new_hypernyms_tail.append(h_id)
                break
        else:
            new_hypernyms_head.append(h_id)
        hypernyms.pop(0)

    new_hypernyms = new_hypernyms_head + new_hypernyms_tail
    assert num_old == len(new_hypernyms), f'new hs have wrong len: {new_hypernyms}'
    return new_hypernyms


if __name__ == "__main__":
    args = parse_args()

    synsets = get_wordnet_synsets(args.wordnet_dir.glob('synsets.*'))
    enrich_with_wordnet_relations(synsets, args.wordnet_dir.glob('synset_relations.*'))

    pred_base_name = '.'.join(args.prediction_path.name.split('.')[:-1])
    reranked_pred_path = args.prediction_path.with_name(pred_base_name + '.rerank.tsv')
    print(f"Writing reranked predictions to {reranked_pred_path}.")
    with open(reranked_pred_path, 'wt') as fout:
        for w_id, hypernyms in get_prediction(args.prediction_path):
            hypernyms = rerank_predictions(hypernyms, synsets)
            for h_id, other in hypernyms:
                senses = ','.join(s['content'] for s in synsets[h_id]['senses'])
                fout.write(f'{w_id}\t{h_id}\t{senses}\n')
