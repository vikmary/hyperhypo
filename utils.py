#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import io
import sys
import json
import ijson
import collections
from typing import Union, List, Dict, Iterator, Callable, Tuple, Any, Set, Optional
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
    phrases = []
    for fp in fpaths:
        sys.stderr.write(f"Parsing {fp}.\n")
        for row in open(fp, 'rt'):
            phr = row.split('\t', 1)[0].strip()
            if phr not in phrases:
                phrases.append(phr)
    return [{'content': phr} for phr in phrases]


def get_train_synsets(fpaths: Iterator[Union[str, Path]],
                      synset_info_fpaths: Iterator[Union[str, Path]]) -> Dict:
    """Gets synsets with id as key, senses and hypernyms as values."""
    # Load mapping from senses 2 synsets
    sense2synsets = collections.defaultdict(set)
    synset2senses = {}
    for fp in synset_info_fpaths:
        sys.stderr.write(f"Parsing {fp}.\n")
        with open(fp, 'rt') as fin:
            # skip header
            fin.readline()
            for ln in fin:
                synset_id, senses = ln.split('\t')[:2]
                senses = [s.strip() for s in senses.split(',')]
                if synset_id not in synset2senses:
                    synset2senses[synset_id] = set(senses)
                    for sense in senses:
                        sense2synsets[sense].add(synset_id)
    # Load hypernyms for each list of senses
    hypernyms2senses = collections.defaultdict(set)
    for fp in fpaths:
        sys.stderr.write(f"Parsing {fp}.\n")
        with open(fp, 'rt') as fin:
            reader = csv.reader(fin, delimiter='\t')
            for row in reader:
                sense, hyper_synset_ids = row[:2]
                hyper_synset_ids = tuple(json.loads(hyper_synset_ids))
                hypernyms2senses[hyper_synset_ids].add(sense)
    # Construct synsets dictionary
    synsets = {}
    for hyper_ids, senses in hypernyms2senses.items():
        synset_ids = set(synset_id
                         for sense in senses
                         for synset_id in sense2synsets[sense]
                         if synset2senses[synset_id].issubset(senses))
        if set(s for s_id in synset_ids for s in synset2senses[s_id]) != senses:
            print(f'Warning: no full synset info for senses {senses}.'
                  f' Found only {synset_ids} synset ids.')
        for synset_id in synset_ids:
            synsets[synset_id] = {
                'senses': [{'content': s} for s in synset2senses[synset_id]],
                'hypernyms': [{'id': h_id} for h_id in hyper_ids]
            }
    return synsets


def get_train_synsets2(fpaths: Iterator[Union[str, Path]]) -> Dict:
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
                senses = [s.strip() for s in senses.split(',')]
                hyper_synset_ids = json.loads(hyper_synset_ids.replace("'", '"'))
                if synset_id not in synsets:
                    synsets[synset_id] = {
                        'senses': [{'content': s} for s in senses],
                        'hypernyms_grouped': [[{'id': i} for i in hyper_synset_ids]]
                    }
                else:
                    synsets[synset_id]['hypernyms_grouped'].append([
                        {'id': i} for i in hyper_synset_ids
                    ])
    return synsets


def get_wordnet_senses(fpaths: Iterator[Union[str, Path]]) -> Dict:
    """Gets senses with id as key and senses as values."""
    senses = {}
    for fp in fpaths:
        sys.stderr.write(f"Parsing {fp}.\n")
        with open(fp, 'rt') as fin:
            xml_parser = BeautifulSoup(fin.read(), "lxml-xml")
        for sense in xml_parser.findAll('sense'):
            sense_d = sense.attrs
            sense_id = sense_d.pop('id')
            senses[sense_id] = sense_d
    return senses


def get_wordnet_synsets(synset_fpaths: Iterator[Union[str, Path]],
                        senses_fpaths: Iterator[Union[str, Path]]) -> Dict:
    """Gets synsets with id as key and senses as values."""
    senses = get_wordnet_senses(senses_fpaths)
    synsets = {}
    for fp in synset_fpaths:
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
                sense_d = senses[sense['id']]
                synset_d['senses'].append({'id': sense['id'],
                                           'lemma': sense_d['lemma'],
                                           'content': sense_d['name']})
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
                    relation_types: List[str] = ('POS-synonymy', 'hypernyms'),
                    related: Optional[Dict[str, int]] = None,
                    level: int = 0) -> Dict[str, int]:
    # print(f'current synset = {synset_id}, level = {level}, related={related}')
    if not related:
        related = {}
    related[synset_id] = level
    for r_type in relation_types:
        for r_synset_d in synsets[synset_id].get(r_type, []):
            if r_type == 'hypernyms':
                new_level = level + 1
            elif r_type == 'POS-synonymy':
                new_level = level
            if r_synset_d['id'] not in related:
                syn_related = get_all_related(r_synset_d['id'],
                                              synsets,
                                              relation_types,
                                              related,
                                              new_level)
                for syn_rel in syn_related:
                    if syn_rel not in related:
                        # print(f'adding {syn_rel} with level {syn_related[syn_rel]}')
                        related[syn_rel] = syn_related[syn_rel]
    return related


def get_cased(s: str, tokenizer: Callable) -> str:
    s = s.lower()
    if ' ' in s:
        return ' '.join(get_cased(token, tokenizer) for token in s.split())
    cands = [s, s.upper(), s.title()]

    cand_lens = [len(tokenizer(c)) for c in cands]
    min_len = min(cand_lens)
    return cands[cand_lens.index(min_len)]


def read_json_by_item(f: io.StringIO) -> Iterator[Tuple[Any, Any]]:
    yield from ijson.kvitems(f, '')


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
