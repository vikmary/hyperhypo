#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
import json
from pathlib import Path

from tqdm import tqdm
from rusenttokenize import ru_sent_tokenize
from rusenttokenize import SHORTENINGS, JOINING_SHORTENINGS, PAIRED_SHORTENINGS

from prepare_corpora.utils import smart_open, count_lines, extract_zip
from prepare_corpora.utils import TextPreprocessor, Sanitizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-path', '-p', type=Path,
                        help='path to a plain text file with corpora')
    parser.add_argument('--max-lines', '-l', type=int, default=None,
                        help='take only first l lines of corpus')
    parser.add_argument('--min-tokens', '-t', type=int, default=4,
                        help='filter sentences with number of tokens less than t')
    parser.add_argument('--news', '-n', action='store_true',
                        help='whether the input corpus is in news format')
    parser.add_argument('--min-characters', '-c', type=int, default=20,
                        help='filter sentences with number of characters less than c')
    parser.add_argument('--preprocessor-model', '-m', default='udpipe',
                        choices=('udpipe', 'regexp-pymorphy'),
                        help='hype of preprocessor model to use')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    preprocessor = TextPreprocessor(args.preprocessor_model,
                                    filter_stresses=False,
                                    filter_empty_brackets=False,
                                    lowercase=True,
                                    lemmatize=True)
    sanitizer = Sanitizer(filter_stresses=True, filter_empty_brackets=True)

    # extract zip files
    if '.zip' in args.data_path.suffixes:
        extracted_data_paths = extract_zip(args.data_path)
    else:
        extracted_data_paths = [args.data_path]

    if args.max_lines is None:
        print(f"Counting number of input lines.")
        num_lines = sum(count_lines(fp) for fp in extracted_data_paths)
    else:
        num_lines = args.max_lines

    i = 0
    pbar = tqdm(total=num_lines, mininterval=10.)

    base_name = args.data_path.name.split('.')[0]
    if args.max_lines is not None:
        base_name += f'-head-{args.max_lines}'
    base_name += f'.{args.preprocessor_model}'
    tokenized_outpath = args.data_path.with_name(f'corpus.{base_name}.token.txt.gz')
    lemmatized_outpath = args.data_path.with_name(f'{base_name}.lemma.txt.gz')
    print(f"Writing tokenized corpus to {tokenized_outpath}.")
    print(f"Writing lemmatized corpus to {lemmatized_outpath}.")
    with smart_open(tokenized_outpath, 'wt') as f_tok, \
            smart_open(lemmatized_outpath, 'wt') as f_lem:
        for fp in extracted_data_paths:
            with smart_open(fp, 'rt') as fin:
                for line in fin:
                    if i >= num_lines:
                        break
                    i += 1
                    pbar.update(1)
                    if args.news:
                        line = line.strip()
                        try:
                            sents = json.loads(line.split(maxsplit=1)[1])
                            sents = [sanitizer(sent) for sent in sents]
                        except:
                            sents = []
                            print("Failed to load sentences.")
                    else:
                        line = sanitizer(line.strip())
                        sents = ru_sent_tokenize(line, SHORTENINGS, JOINING_SHORTENINGS,
                                                 PAIRED_SHORTENINGS)
                    for sent in sents:
                        try:
                            sent = sent.replace('\n', '')
                            if len(sent) < args.min_characters:
                                continue
                            tokens, sent_prep = preprocessor(sent)
                            if len(tokens) < args.min_tokens:
                                continue
                        except Exception as msg:
                            print(f"WARNING: error {msg} for sent = {sent}")
                            continue
                        sent_tokenized = ' '.join(tokens)
                        if len(sent_tokenized.split()) != len(sent_prep.split()):
                            print("Tokenized and lemmatized texts have different lengths"
                                  ", skipping.")
                            continue
                        f_tok.write(sent_tokenized + '\n')
                        f_lem.write(sent_prep + '\n')
    pbar.close()
