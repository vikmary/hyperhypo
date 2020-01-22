#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from typing import Union, List, Dict, Iterator
from pathlib import Path
import csv

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
    synsets = {}
    for fp in fpaths:
        sys.stderr.write(f"Parsing {fp}.\n")
        with open(fp, 'rt') as fin:
            reader = csv.reader(fin, delimiter='\t')
            next(reader)
            for row in reader:
                synset_id, senses, hyper_synset_ids = row[:3]
                synset_description = ''
                if len(row) > 3:
                    synset_description = row[3]
                senses = senses.split(',')
                hyper_synset_ids = hyper_synset_ids.split(',')
                if synset_id in synsets:
                    raise ValueError(f"multiple synsets with id = \'{synset_id}\'")
                synsets[synset_id] = {'senses': [{'content': s} for s in senses],
                                           'description': synset_description,
                                           'hypernyms': [{'id': i}
                                                         for i in hyper_synset_ids]}
    return synsets


def get_wordnet_synsets(fpaths: Iterator[Union[str, Path]]) -> Dict:
    """Gets synsets with id as key and senses as values."""
    synsets = {}
    for fp in fpaths:
        sys.stderr.write(f"Parsing {fp}.\n")
        with open(fp, 'rt') as fin:
            xml_parser = BeautifulSoup(fin.read(), "lxml-xml")
        for synset in xml_parser.findAll('synset'):
            synset_d = synset.attrs
            synset_id = synset_d.pop('id')
            if synset_id in synsets:
                raise ValueError(f"multiple synsets with id = \'{synset_id}\'")
            # finding child senses
            synset_d['senses'] = []
            for sense in synset.findAll('sense'):
                synset_d['senses'].append({'id': sense.get('id'),
                                           'content': sense.contents[0]})
            # adding to dict of synsets
            synsets[synset_id] = synset_d
    return synsets


def enrich_with_wordnet_relations(synsets: dict,
                                  relation_fpaths: Iterator[Union[str, Path]]) -> None:
    for fp in relation_fpaths:
        sys.stderr.write(f"Parsing {fp}.\n")
        with open(fp, 'rt') as fin:
            xml_parser = BeautifulSoup(fin.read(), "lxml-xml")
        for relation in xml_parser.findAll('relation'):
            attrs = relation.attrs
            relation_type = attrs['name']
            if relation_type in ('domain', 'part holonym', 'part meronym',
                                 'cause', 'entailment'):
                continue
            elif relation_type in ('instance hyponym', 'hyponym'):
                hypernym_id, hyponym_id = attrs['parent_id'], attrs['child_id']
                if 'hypernyms' not in synsets[hyponym_id]:
                    synsets[hyponym_id]['hypernyms'] = []
                if hypernym_id not in synsets[hyponym_id]['hypernyms']:
                    synsets[hyponym_id]['hypernyms'].append(hypernym_id)
            elif relation_type in ('instance hypernym', 'hypernym'):
                hypernym_id, hyponym_id = attrs['child_id'], attrs['parent_id']
                if 'hypernyms' not in synsets[hyponym_id]:
                    synsets[hyponym_id]['hypernyms'] = []
                if hypernym_id not in synsets[hyponym_id]['hypernyms']:
                    synsets[hyponym_id]['hypernyms'].append(hypernym_id)
            elif relation_type in ('antonym', 'POS-synonymy'):
                pair = attrs['child_id'], attrs['parent_id']
                if pair[1] not in synsets[pair[0]].get(relation_type, []):
                    synsets[pair[0]][relation_type] = \
                        synsets[pair[0]].get(relation_type, []) + [pair[1]]
                if pair[0] not in synsets[pair[1]].get(relation_type, []):
                    synsets[pair[1]][relation_type] = \
                        synsets[pair[1]].get(relation_type, []) + [pair[0]]
            else:
                raise ValueError(f"invalid relation name '{relation_type}'")


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
    data_path = Path(sys.argv[1])

    synsets = get_wordnet_synsets(data_path.glob('synsets.*'))
    enrich_with_wordnet_relations(synsets, data_path.glob('synset_relations.*'))

    sys.stderr.write(f"Found {len(synsets)} synsets.\n")

    sys.stderr.write(f"\nExamples:\n")
    for i, s in enumerate(synsets.items()):
        if i > 2:
            break
        sys.stderr.write(f'{s}\n')
