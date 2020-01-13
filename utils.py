#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from typing import Union, List, Dict, Iterator
from pathlib import Path

from bs4 import BeautifulSoup


def synsets2senses(synsets: dict) -> List[dict]:
    """Converts dict of synsets to senses."""
    senses = []
    for synset in synsets.values():
        for s in synset['senses']:
            senses.append(s)
            if 'hypernyms' in s:
                senses[-1]['hypernyms'] = s['hypernyms']
    return senses


def get_test_senses(fpaths: Iterator[Union[str, Path]]) -> List[dict]:
    """Gets test senses with their texts."""
    senses = []
    for fp in fpaths:
        sys.stderr.write(f"Parsing {fp}.\n")
        for row in open(fp, 'rt'):
            senses.append({'content': row.strip()})
    return senses


def get_train_synsets(fpaths: Iterator[Union[str, Path]]) -> Dict:
    """Gets synsets with id as key, senses and hyperonyms as values."""
    synsets_dict = {}
    for fp in fpaths:
        sys.stderr.write(f"Parsing {fp}.\n")
        with open(fp, 'rt') as fin:
            fin.readline()
            for row in fin:
                synset_id, senses, hyper_synset_ids = row.rstrip().split('\t', 2)
                senses = senses.split(',')
                hyper_synset_ids = hyper_synset_ids.split(',')
                if synset_id in synsets_dict:
                    raise ValueError(f"multiple synsets with id = \'{synset_id}\'")
                synsets_dict[synset_id] = {'senses': [{'content': s} for s in senses],
                                           'hypernyms': [{'id': i}
                                                         for i in hyper_synset_ids]}
    return synsets_dict


def get_synsets(fpaths: Iterator[Union[str, Path]]) -> Dict:
    """Gets synsets with id as key and senses as values."""
    synsets_dict = {}
    for fp in fpaths:
        sys.stderr.write(f"Parsing {fp}.\n")
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
                synset_d['senses'].append({'id': sense.get('id'),
                                           'content': sense.contents[0]})
            # adding to dict of synsets
            synsets_dict[synset_id] = synset_d
    return synsets_dict


def get_hyperstar_senses(fpaths: Iterator[Union[str, Path]]) -> List[dict]:
    """Gets senses with their hypernyms."""
    senses = []
    for fp in fpaths:
        sys.stderr.write(f"Parsing {fp}.\n")
        with open(fp, 'rt') as fin:
            for row in fin:
                hyponym, hypernym = row.rstrip().split()
                if senses and (hyponym == senses[-1]['content']):
                    senses[-1]['hypernyms'].append({'content': hypernym})
                else:
                    senses.append({'content': hypernym,
                                   'hypernyms': [{'content': hypernym}]})
    return senses

if __name__ == "__main__":
    synsets = get_synsets(sys.argv[1:])
    sys.stderr.write(f"Found {len(synsets)} synsets.\n")

    sys.stderr.write(f"\nExamples:\n")
    for i, s in enumerate(synsets.items()):
        if i > 2:
            break
        sys.stderr.write(f'{s}\n')
