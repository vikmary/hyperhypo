#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
from typing import Union, List, Dict, Iterator, Callable
from pathlib import Path
import random
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
            word = row.split('\t', 1)[0].strip()
            senses.append({'content': word})
    return senses


def get_train_synsets(fpaths: Iterator[Union[str, Path]]) -> Dict:
    """Gets synsets with id as key, senses and hyperonyms as values."""
    # TODO: add new function enrich_with_wordnet_senses
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
                hypernym = {'id': hypernym_id}
                if hypernym not in synsets[hyponym_id]['hypernyms']:
                    synsets[hyponym_id]['hypernyms'].append(hypernym)
            elif relation_type in ('instance hypernym', 'hypernym'):
                hypernym_id, hyponym_id = attrs['child_id'], attrs['parent_id']
                if 'hypernyms' not in synsets[hyponym_id]:
                    synsets[hyponym_id]['hypernyms'] = []
                hypernym = {'id': hypernym_id}
                if hypernym not in synsets[hyponym_id]['hypernyms']:
                    synsets[hyponym_id]['hypernyms'].append(hypernym)
            elif relation_type in ('antonym', 'POS-synonymy'):
                w0, w1 = attrs['child_id'], attrs['parent_id']
                if {'id': w0} not in synsets[w1].get(relation_type, []):
                    synsets[w1][relation_type] = \
                        synsets[w1].get(relation_type, []) + [{'id': w0}]
                if {'id': w1} not in synsets[w0].get(relation_type, []):
                    synsets[w0][relation_type] = \
                        synsets[w0].get(relation_type, []) + [{'id': w1}]
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


def get_all_related(synset_id: str,
                    synsets: Dict[str, dict],
                    relation_types: List[str]=['POS_synonymy', 'hypernyms']) -> List[str]:
    related = [synset_id]
    for r_type in relation_types:
        for r_synset_d in synsets[synset_id].get(r_type, []):
            if r_synset_d['id'] not in related:
                related.extend(get_all_related(r_synset_d['id'], synsets, relation_types))
    return list(set(related))


def get_cased(s: str, tokenizer: Callable) -> str:
    s = s.lower()
    if ' ' in s:
        return ' '.join(get_cased(token, tokenizer) for token in s.split())
    cands = [s, s.upper(), s.title()]

    cand_lens = [len(tokenizer(c)) for c in cands]
    min_len = min(cand_lens)
    return cands[cand_lens.index(min_len)]


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

    synset_id = random.choice(list(synsets.keys()))
    sys.stderr.write(f"\nSynsets related to {synset_id}"
                     f" ({synsets[synset_id]['ruthes_name']}):\n")
    for i, s_id in enumerate(get_all_related(synset_id, synsets)):
        sys.stderr.write(f'{i+1}. {synsets[s_id]}\n')
