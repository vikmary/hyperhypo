#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv
from pathlib import Path

from utils import get_wordnet_synsets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--wordnet-dir', '-w', type=Path,
                        help='path to a wordnet directory')
    parser.add_argument('--output_path', '-o', type=Path,
                        help='output file with candidates')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    synsets = get_wordnet_synsets(args.wordnet_dir.glob('synsets.*'))

    with open(args.output_path, 'wt') as fout:
        writer = csv.writer(fout, delimiter='\t', quoting=csv.QUOTE_MINIMAL,
                            quotechar='"')

        for synset_id, synset in synsets.items():
            for sense in synset['senses']:
                writer.writerow([sense['content'].lower(),
                                 synset_id,
                                 synset['ruthes_name'].lower()])
