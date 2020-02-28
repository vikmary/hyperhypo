#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import collections
from pathlib import Path
from tqdm import tqdm

from utils import enrich_with_wordnet_relations, get_cased, get_wordnet_synsets
from prepare_corpora.utils import TextPreprocessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wordnet-dir', '-w', type=Path,
                        help='path to a wordnet directory')
    parser.add_argument('--bert-model-path', '-b', type=Path, required=False,
                        help='path to a bert model path to be used for case detection')
    parser.add_argument('--output_path', '-o', type=Path,
                        help='output file with candidates')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    preprocessor = TextPreprocessor('regexp+pymorphy',
                                    filter_empty_brackets=True,
                                    filter_stresses=True,
                                    lowercase=True,
                                    lemmatize=False)

    synsets = get_wordnet_synsets(args.wordnet_dir.glob('synsets.*'),
                                  args.wordnet_dir.glob('senses.*'))
    enrich_with_wordnet_relations(synsets, args.wordnet_dir.glob('synset_relations.*'))

    if args.bert_model_path is not None:
        from transformers import BertTokenizer
        tok = BertTokenizer.from_pretrained(args.bert_model_path,
                                            do_lower_case=False).tokenize
        print("Using bert tokenizer to detect case of words.")

    hypernym2hyponyms = collections.defaultdict(set)
    for synset_id, synset in tqdm(synsets.items()):
        for h in synset.get('hypernyms', []):
            hypernym2hyponyms[h['id']].add(synset_id)

    train_dict = {}
    for hyper_id, hypo_ids in hypernym2hyponyms.items():
        synonyms = [[s['content'] for s in synsets[syn_id]['senses']] +
                    [synsets[syn_id]['ruthes_name']]
                    for syn_id in hypo_ids]
        if args.bert_model_path is not None:
            synonyms = [[get_cased(preprocessor(s)[1], tok) for s in syn_list]
                        for syn_list in synonyms]
        train_dict[hyper_id] = synonyms

    print(f"Writing output json to {args.output_path}.")
    json.dump(train_dict, open(args.output_path, 'wt'), indent=2, ensure_ascii=False)

