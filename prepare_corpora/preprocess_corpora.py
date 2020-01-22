#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import re
from pathlib import Path

from tqdm import tqdm
from rusenttokenize import ru_sent_tokenize
from rusenttokenize import SHORTENINGS, JOINING_SHORTENINGS, PAIRED_SHORTENINGS

from utils import smart_open, count_lines, extract_zip, Sanitizer, Lemmatizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-paths', '-p', type=Path, nargs='+',
                        help='paths to a plain text files with corpora')
    parser.add_argument('--max-lines', '-l', type=int, default=None,
                        help='take only first l lines of corpus')
    parser.add_argument('--min-tokens', '-t', type=int, default=4,
                        help='filter sentences with number of tokens less than t')
    parser.add_argument('--min-characters', '-c', type=int, default=20,
                        help='filter sentences with number of characters less than c')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    tokenizer = re.compile(r"[\w']+|[^\w ]")
    sanitizer = Sanitizer(filter_diacritical=True, filter_empty_brackets=True)
    lemmatizer = Lemmatizer()

    # extract zip files
    data_paths = []
    for fp in args.data_paths:
        if '.zip' in fp.suffixes:
            extracted_fpaths = extract_zip(fp)
            data_paths.extend(extracted_fpaths)
        else:
            data_paths.append(fp)

    if args.max_lines is None:
        print(f"Counting number of input lines.")
        num_lines = sum(count_lines(fp) for fp in args.data_paths)
    else:
        num_lines = args.max_lines

    i = 0
    pbar = tqdm(total=num_lines, mininterval=10.)
    for fp in data_paths:
        if i >= num_lines:
            break
        base_name = fp.name.split('.')[0]
        tokenized_outpath = fp.with_name(base_name + '.token.txt.gz')
        lemmatized_outpath = fp.with_name(base_name + '.lemma.txt.gz')
        print(f"Writing tokenized corpus to {tokenized_outpath}.")
        print(f"Writing lemmatized corpus to {lemmatized_outpath}.")

        with smart_open(fp, 'rt') as fin, \
                smart_open(tokenized_outpath, 'wt') as f_tok, \
                smart_open(lemmatized_outpath, 'wt') as f_lem:
            for line in fin:
                if i >= num_lines:
                    break
                i += 1
                pbar.update(1)
                line = sanitizer(line.strip())
                if line:
                    for sent in ru_sent_tokenize(line, SHORTENINGS, JOINING_SHORTENINGS,
                                                 PAIRED_SHORTENINGS):
                        try:
                            if len(sent) < args.min_characters:
                                continue
                            tokens = tokenizer.findall(sent)
                            if len(tokens) < args.min_tokens:
                                continue
                            lemmas = [lemmatizer(t) for t in tokens]
                        except Exception as msg:
                            print(f"WARNING: error {msg} for sent = {sent}")
                            continue
                        f_tok.write(' '.join(tokens) + '\n')
                        f_lem.write(' '.join(lemmas) + '\n')
    pbar.close()
