#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
from pathlib import Path
from tqdm import tqdm

from utils import get_train_synsets, get_wordnet_synsets, enrich_with_wordnet_relations
from utils import get_cased
from prepare_corpora.utils import TextPreprocessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-paths', '-t', type=Path, nargs='+',
                        help='path(s) to training data')
    parser.add_argument('--wordnet-dir', '-w', type=Path,
                        help='path to a wordnet directory')
    parser.add_argument('--bert-model-path', '-b', type=Path, required=False,
                        help='path to a bert model path to be used for case detection')
    parser.add_argument('--output_path', '-o', type=Path,
                        help='output file with candidates')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    preprocessor = TextPreprocessor(filter_empty_brackets=True,
                                    filter_stresses=True,
                                    lowercase=True,
                                    lemmatize=True)

    train_synsets = get_train_synsets(args.data_paths)

    synsets = get_wordnet_synsets(args.wordnet_dir.glob('synsets.*'))
    enrich_with_wordnet_relations(synsets, args.wordnet_dir.glob('synset_relations.*'))

    if args.bert_model_path is not None:
        from transformers import BertTokenizer
        tok = BertTokenizer.from_pretrained(args.bert_model_path,
                                            do_lower_case=False).tokenize
        print("Using bert tokenizer to detect case of words.")

    train_dict = {}
    for synset_id, synset in tqdm(train_synsets.items()):
        senses = [s['content'] for s in synset['senses']]
        # construct hypernyms
        hypernym_ids = [h['id'] for h in synset['hypernyms']]
        hypernyms = [[s['content'] for s in synsets[h_id]['senses']]
                     for h_id in hypernym_ids]

        if args.bert_model_path is not None:
            senses = [get_cased(preprocessor(s), tok) for s in senses]
            hypernyms = [[get_cased(preprocessor(h), tok) for h in h_list]
                         for h_list in hypernyms]
        for sense in senses:
            train_dict[sense] = (senses, hypernyms)

    print(f"Writing output json to {args.output_path}.")
    json.dump(train_dict, open(args.output_path, 'wt'), indent=2, ensure_ascii=False)
