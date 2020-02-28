#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import argparse
import collections
from pathlib import Path
from tqdm import tqdm

from utils import get_train_synsets, get_wordnet_synsets, enrich_with_wordnet_relations
from utils import get_cased, get_wordnet_senses
from prepare_corpora.utils import TextPreprocessor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-paths', '-t', type=Path, nargs='+',
                        help='path(s) to training data')
    parser.add_argument('--synset-info-paths', '-s', type=Path, nargs='+',
                        help='paths to synset info for training data')
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

    train_synsets = get_train_synsets(args.data_paths, args.synset_info_paths)

    all_senses = get_wordnet_senses(args.wordnet_dir.glob('senses.*'))
    all_contents = set(s['name'] for s in all_senses.values())
    for i, synset in enumerate(train_synsets.values()):
        for j, sense in enumerate(synset['senses']):
            if sense['content'] not in all_contents:
                all_contents.add(sense['content'])
                all_senses[f'train_{i}_{j}_0'] = {'name': sense['content']}

    synsets = get_wordnet_synsets(args.wordnet_dir.glob('synsets.*'),
                                  args.wordnet_dir.glob('senses.*'))
    enrich_with_wordnet_relations(synsets, args.wordnet_dir.glob('synset_relations.*'))

    converted = 0
    for s_id in train_synsets:
        if sorted(s['content'] for s in train_synsets[s_id]['senses']) != \
                sorted(s['content'] for s in synsets[s_id]['senses']):
            train_synsets[s_id]['senses'] = synsets[s_id]['senses']
            converted += 1
    print(f"Found {converted}/{len(train_synsets)} synsets from train that"
          f" do not match wordnet synsets, replaced them with wordnet synsets.")

    if args.bert_model_path is not None:
        from transformers import BertTokenizer
        tok = BertTokenizer.from_pretrained(args.bert_model_path,
                                            do_lower_case=False).tokenize
        print("Using bert tokenizer to detect case of words.")

    train_dict = collections.defaultdict(list)
    for synset_id, synset in tqdm(train_synsets.items()):
        senses = sorted([s['content'] for s in synset['senses']])  # + [synset['ruthes_name']]
        # construct hypernyms
        hypernym_ids = [h['id'] for h in synset['hypernyms']]
        hypernyms = [sorted([s['content'] for s in synsets[h_id]['senses']]) +
                     []  # [synsets[h_id]['ruthes_name']]
                     for h_id in hypernym_ids]

        if args.bert_model_path is not None:
            senses = [get_cased(preprocessor(s)[1], tok) for s in senses]
            hypernyms = [[get_cased(preprocessor(h)[1], tok) for h in h_list]
                         for h_list in hypernyms]
        for sense in senses:
            train_dict[sense].append((senses, hypernyms))

    print(f"Writing output json to {args.output_path}.")
    json.dump(train_dict, open(args.output_path, 'wt'), indent=2, ensure_ascii=False)

    synset_info = {}
    for s_id, synset in synsets.items():
        senses = sorted([preprocessor(s['content'])[1].upper() for s in synset['senses']])
        synset_info[tuple(senses)] = (synset['ruthes_name'], s_id)

    print(f"Writing to synset_info.not_lemma.json.")
    json.dump([[senses, info[0]] for senses, info in synset_info.items()],
              open(args.output_path.with_name('synset_info.not_lemma.json'), 'wt'),
              indent=2,
              ensure_ascii=False)

    print(f"Writing to synset_info_extended.not_lemma.json.")
    json.dump([[senses, info[0], info[1]] for senses, info in synset_info.items()],
              open(args.output_path.with_name('synset_info_extended.not_lemma.json'), 'wt'),
              indent=2,
              ensure_ascii=False)

    lemma_path = args.output_path.with_name('lemmatized_info.json')
    print(f"Writing lemma info to {lemma_path}.")
    for i, synset in enumerate(train_synsets.values()):
        for j, sense in enumerate(synset['senses']):
            if sense['content'] not in all_contents:
                all_contents.add(sense['content'])
                all_senses[f'train_{i}_{j}_1'] = {'name': sense}
    preprocessor = TextPreprocessor('regexp+pymorphy',
                                    filter_empty_brackets=True,
                                    filter_stresses=True,
                                    lowercase=True,
                                    lemmatize=True)
    norm_sense2senses = collections.defaultdict(list)
    for sense in all_senses.values():
        canonical_phr = sense.get('content', sense['name'])
        for phr in [sense.get('content'), sense.get('lemma'), sense.get('name')]:
            if not phr:
                continue
            phr = phr.lower()
            if canonical_phr not in norm_sense2senses[phr]:
                norm_sense2senses[phr].append(canonical_phr)
            phr_norm = preprocessor(phr)[1]
            if canonical_phr not in norm_sense2senses[phr_norm]:
                norm_sense2senses[phr_norm].append(canonical_phr)
    json.dump(norm_sense2senses, open(lemma_path, 'wt'), indent=2, ensure_ascii=False)

    for sense in train_dict:
        for senses, hypernyms in train_dict[sense]:
            for synset in [senses] + hypernyms:
                synset_upper = tuple(s.upper() for s in synset)
                if synset_upper not in synset_info:
                    print(f"Warning! senses {synset_upper} not found in synset_info.")
