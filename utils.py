#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from typing import Union, List, Dict, Iterator
from pathlib import Path

from bs4 import BeautifulSoup


def get_synsets(fpaths: Iterator[Union[str, Path]]) -> Dict:
    """Gets synsets with id as key and senses as values."""
    synsets_dict = {}
    for fp in fpaths:
        print(f"Parsing {fp}.")
        xml_data = open(fp, 'rt').read()
        xml_parser = BeautifulSoup(xml_data, "lxml-xml")
        for synset in xml_parser.findAll('synset'):
            synset_d = synset.attrs
            synset_id = synset_d.pop('id')
            if synset_id in synsets_dict:
                raise ValueError(f"multiple synsets with id = \'{synset_id}\'")
            # finding child senses
            synset_d['senses'] = []
            for sense in synset.findAll('sense'):
                synset_d['senses'].append({'id': sense.get('id')})
            # adding to dict of synsets
            synsets_dict[synset_id] = synset_d
    return synsets_dict


if __name__ == "__main__":
    synsets = get_synsets(sys.argv[2:])
    print(f"Found {len(synsets)} synsets.")

    print(f"\nExamples:")
    for i, s in enumerate(synsets.items()):
        if i > 2:
            break
        print(s)

