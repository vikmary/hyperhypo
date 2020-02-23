#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import json
from pathlib import Path

from corpus_indexed import CorpusIndexed
from utils import get_train_synsets
from prepare_corpora.utils import TextPreprocessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', '-d', type=Path,
                        help='path to lemmatized corpora')
    parser.add_argument('--train-paths', '-t', type=Path, nargs='+',
                        help='path(s) to text file(s) with training data')
    parser.add_argument('--out-prefix', type=str,
                        help='prefix of the output file')
    parser.add_argument('--synset-info-paths', '-s', type=Path, nargs='+',
                        help='paths to synset info for training data')
    parser.add_argument('--max-lines', '-l', type=int, default=None,
                        help='take only first l lines of corpus')
    parser.add_argument('--max-lines-per-item', '-m', type=int, default=300,
                        help='max utterances for each word in index')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # TODO: add parsing udpipe suffix
    preprocessor = TextPreprocessor('regexp+pymorphy',
                                    filter_stresses=True,
                                    filter_empty_brackets=True,
                                    lowercase=True,
                                    lemmatize=True)

    hyponyms = set()
    for synset_id, synset in get_train_synsets(args.train_paths, args.synset_info_paths)\
            .items():
        for sense in synset['senses']:
            _, hypo = preprocessor(sense['content'])
            if not hypo:
                print(f"Skipping hyponym {hypo}, because it's lemma is empty.")
            else:
                hyponyms.add(hypo)

    inverted_index = CorpusIndexed.build_index(args.data_path,
                                               vocab=hyponyms,
                                               max_utterances=args.max_lines,
                                               max_utterances_per_item=args.max_lines_per_item)

    base_name = args.data_path.name.split('.')[0]
    if args.max_lines is not None:
        base_name += f'-head-{args.max_lines}'
    out_path = args.data_path.with_name(args.out_prefix + base_name + '.json')

    print(f"Writing training index to {out_path}.")
    json.dump(inverted_index, open(out_path, 'wt'), indent=2, ensure_ascii=False)
