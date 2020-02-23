#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import json
from pathlib import Path
from typing import List, Dict
from urllib.parse import quote
from tqdm import tqdm
from multiprocessing import Pool
import requests
import re

from lxml import html


N_WORKERS = 30


def filter_intro_abbreviations(word: str) -> str:
    elements = re.split('^(\w{1,6}\.,?\s?)+', word)
    if len(elements) > 1:
        return elements[-1].strip()
    else:
        return elements[0].strip()


def get_word_definitions(word: str, filter_abbr: bool=True) -> Dict[str, List[str]]:
    title = quote(word)
    definitions_xpath = '/html/body/div[3]/div[3]/div[4]/div/ol[1]/li'
    url = f'https://ru.wiktionary.org/w/index.php?title={title}&printable=yes'
    page = requests.get(url)
    root = html.fromstring(page.text)
    tree = root.getroottree()
    examples_sep = '◆'
    definitions = []
    for el in tree.xpath(definitions_xpath):
        text = el.text_content()
        definition, *examples = text.split(examples_sep)
        if filter_abbr:
            definition = filter_intro_abbreviations(definition)
        definition = definition.replace('\xa0', ' ')
        definitions.append(definition)
    return word, definitions


def get_definitions(words: List[str]) -> Dict[str, List[str]]:
    with Pool(N_WORKERS) as p:
        result = list(tqdm(p.imap(get_word_definitions, words), total=len(words)))
    definitions_dict = {}
    for word, definitions in result:
        definitions_dict[word] = definitions
    return definitions_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path,
                        help='JSON file with dictionary, inference for'
                             ' keys of the dict',
                        default='/home/hdd/data/hypernym/train.cased.json')
    parser.add_argument('-o', '--output', type=Path,
                        help='Output file with definitions, '
                             'JSON dict {word_0: [def_0, def_1, ...],'
                             'word_1: [def_0, def_1, ...], ...}',
                        default='/home/hdd/data/hypernym/definitions.train.json')
    args = parser.parse_args()
    with open(args.input) as fin:
        hyponyms = list(json.load(fin))

    definitions = get_definitions(hyponyms)
    with open(args.output, 'w', encoding='utf-8') as fin:
        json.dump(definitions, fin, indent=4, ensure_ascii=False)

