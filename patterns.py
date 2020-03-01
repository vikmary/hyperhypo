#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import tqdm
import collections
from pathlib import Path
from typing import Union, Dict, List, Tuple, Optional

from utils import get_test_senses, get_wordnet_synsets, enrich_with_wordnet_relations
from prepare_corpora.utils import TextPreprocessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pattern-pairs', type=Path,
                        help='path to a file with hyponym-hypernym pairs'
                        ' extracted from text.')
    parser.add_argument('--data-path', type=Path,
                        help='path to a file with hyponym phrases')
    parser.add_argument('--wordnet-dir', type=Path, required=True,
                        help='path to a wordnet directory')
    parser.add_argument('--outpath', type=Path, required=True,
                        help='path for output predictions')
    parser.add_argument('--pos', type=Optional[str], default=None,
                        choices=(None, 'nouns', 'verbs', 'adjectives'),
                        help='filter hypernyms of only this type of pos')
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
    test_hyper_phrs = []
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
        test_hyper_phrs.append(hyper_phrs)

    found = len(test_phrs) - skipped
    print(f"Found hyper candidate phrases for {found}/{len(test_phrs)}"
          f" ({int(found/len(test_phrs) * 100)}%) test phrases.")

    synsets = get_wordnet_synsets(args.wordnet_dir.glob('synsets.*'),
                                  args.wordnet_dir.glob('senses.*'))
    enrich_with_wordnet_relations(synsets, args.wordnet_dir.glob('synset_relations.*'))

    sense2synsets = collections.defaultdict(list)
    for s_id, synset in synsets.items():
        for sense in synset['senses']:
            sense2synsets[sense['content'].lower()].append(s_id)
        sense2synsets[synset['ruthes_name'].lower()].append(s_id)

    norm_sense2senses = collections.defaultdict(list)
    for sense in sense2synsets:
        norm_sense2senses[preprocessor(sense)[1]].append(sense)

    skipped = 0
    print(f"Looking for synsets of extracted hypernym phrases.")
    print(f"Writing prediction to {args.outpath}.")
    with open(args.outpath, 'wt') as fout:
        for test_phr, hyper_phrs in zip(test_phrs, test_hyper_phrs):
            hyper_synsets = {}
            for phr, freq in hyper_phrs:
                if phr in sense2synsets:
                    for s_id in sense2synsets[phr]:
                        hyper_synsets[s_id] = hyper_synsets.get(s_id, 0) + freq
                else:
                    _, phr_norm = preprocessor(phr)
                    if phr_norm in norm_sense2senses:
                        for sense in norm_sense2senses[phr_norm]:
                            for s_id in sense2synsets[sense]:
                                hyper_synsets[s_id] = hyper_synsets.get(s_id, 0) + freq
            if args.pos:
                hyper_synsets = {s_id: freq
                                 for s_id, freq in hyper_synsets.items()
                                 if s_id[-1].lower() == args.pos[0]}
            if not hyper_synsets:
                skipped += 1
            else:
                sum_freq = sum(hyper_synsets.values())
                for s_id, freq in sorted(hyper_synsets.items(), key=lambda x: x[1], reverse=True):
                    prob = round(freq / sum_freq, 3)
                    senses_str = ','.join(s['content'] for s in synsets[s_id]['senses'])
                    fout.write(f"{test_phr.upper()}\t{s_id}\t{prob}\t{senses_str}\n")

    found = len(test_phrs) - skipped
    print(f"Found hyper synsets for {found}/{len(test_phrs)}"
          f" ({int(found/len(test_phrs) * 100)}%) test phrases.")

