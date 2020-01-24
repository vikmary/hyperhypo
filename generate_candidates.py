#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path
from tqdm import tqdm

from utils import get_wordnet_synsets, get_cased


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
    synsets = get_wordnet_synsets(args.wordnet_dir.glob('synsets.*'))

    if args.bert_model_path is not None:
        from transformers import BertTokenizer
        tokenizer = BertTokenizer.from_pretrained(args.bert_model_path,
                                                  do_lower_case=False).tokenize
        print("Using bert tokenizer to detect case of words.")

    print(f"Writing candidates to {args.output_path}.")
    with open(args.output_path, 'wt') as fout:
        writer = csv.writer(fout, delimiter='\t', quoting=csv.QUOTE_MINIMAL,
                            quotechar='"')

        for synset_id, synset in tqdm(synsets.items()):
            for sense in synset['senses']:
                sense_name = sense['content'].lower()
                ruthes_name = synset['ruthes_name'].lower()
                if args.bert_model_path is not None:
                    sense_name = get_cased(sense_name, tokenizer=tokenizer)
                    ruthes_name = get_cased(ruthes_name, tokenizer=tokenizer)

                writer.writerow([sense_name, synset_id, ruthes_name])
