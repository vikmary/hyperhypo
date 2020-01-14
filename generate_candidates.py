#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import csv

from utils import get_synsets


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('synset_paths', nargs='+',
                        help='xml files with synset data')
    parser.add_argument('output_path', help='output file with candidates')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    synsets = get_synsets(args.synset_paths)

    with open(args.output_path, 'wt') as fout:
        writer = csv.writer(fout, delimiter='\t', quoting=csv.QUOTE_MINIMAL, quotechar='"')

        for synset_id, synset in synsets.items():
            writer.writerow([synset_id,
                             synset['ruthes_name'].replace('\t', ' '),
                             ', '.join(s['content'].replace('\t', ' ')
                                       for s in synset['senses'])])
