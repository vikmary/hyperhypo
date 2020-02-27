#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import random
import json
import re
from collections import Counter
from pathlib import Path
from typing import List, Dict
from urllib.parse import quote

from ru_sent_tokenize import ru_sent_tokenize
from tqdm import tqdm
from multiprocessing import Pool
import requests
import wikipedia
from lxml import html

from prepare_corpora.utils import Sanitizer

wikipedia.set_lang('ru')
N_WORKERS = 1


class DefinitionDB:
    def __init__(self,
                 db_path: str='/home/hdd/data/hypernym/wiki_wikt_db.json',
                 filter_abbr: bool = True,
                 lowercase: bool = True,
                 filter_wiki_brackets = True):
        self.db_path = Path(db_path)
        self.filter_abbr = filter_abbr
        self.lowercase = lowercase
        self._examples_sep = '◆'
        self.filter_wiki_brackets = filter_wiki_brackets

        if not self.db_path.exists():
            with self.db_path.open('w') as fin:
                json.dump({'wiki': {}, 'wikt': {}}, fin)
        with self.db_path.open('r') as fin:
            db = json.load(fin)
            self.wiki_db = db['wiki']
            self.wikt_db = db['wikt']
            self.wiki_db: Dict[str, List[str]]
            self.wikt_db: Dict[str, List[str]]
        self.san = Sanitizer()

    @staticmethod
    def filter_intro_abbreviations(word: str) -> str:
        elements = re.split('^(\w{1,6}\.,?\s?)+', word)
        if len(elements) > 1:
            return elements[-1].strip()
        else:
            return elements[0].strip()

    def get_word_definitions(self, word: str) -> List[str]:
        if word in self.wikt_db:
            definitions = self.wikt_db[word]
        if (word in self.wikt_db and not self._any_valid_defs(definitions)) or word not in self.wikt_db:
            title = quote(word)
            definitions_xpath = '/html/body/div[3]/div[3]/div[4]/div/ol[1]/li'
            url = f'https://ru.wiktionary.org/w/index.php?title={title}&printable=yes'
            page = requests.get(url)
            root = html.fromstring(page.text)
            tree = root.getroottree()
            definitions = []
            for el in tree.xpath(definitions_xpath):
                definition = el.text_content()
                definition = definition.replace('\xa0', ' ').strip()
                print(f'WIKT {word}: {definition}')
                if definition:
                    definitions.append(definition)
        return definitions

    def _any_valid_defs(self, defs: List[str]) -> bool:
        return any(d.strip() for d in defs)

    def get_defs(self, word: str) -> List[str]:
        return self.get_word_definitions(word), self.get_definition_wikipedia(word)

    def get_definitions(self, words: List[str]) -> List[List[str]]:
        with Pool(N_WORKERS) as p:
            defs = list(p.imap(self.get_defs, words))
            definitions_wikt, definitions_wiki = list(zip(*defs))
        return definitions_wikt, definitions_wiki

    def get_definition_wikipedia(self, word):
        if word in self.wiki_db:
            definitions = self.wiki_db[word]
        if (word in self.wiki_db and not self._any_valid_defs(definitions)) or word not in self.wiki_db:
            try:
                definitions = [wikipedia.summary(word.lower())]
            except wikipedia.DisambiguationError as e:
                c = Counter(e.options)
                options = [key for key, val in c.items() if val == 1]

                try:
                    s = random.choice(options)
                    definitions = [wikipedia.summary(s)]
                except:
                    definitions = []
            except:
                definitions = []
        return definitions

    def get_definition(self, words: List[str]) -> List[List[str]]:
        if self.lowercase:
            words = [word.lower() for word in words]
        filtered_definitions = []
        for n in tqdm(range(0, len(words), 1000)):
            current_words = words[n: n+1000]
            definitions_wikt, definitions_wiki = self.get_definitions(current_words)
            self.wiki_db.update({word: definition for word, definition in zip(current_words, definitions_wiki) if definition})
            self.wikt_db.update({word: definition for word, definition in zip(current_words, definitions_wikt) if definition})

            with self.db_path.open('w') as fin:
                json.dump({'wiki': self.wiki_db, 'wikt': self.wikt_db}, fin, ensure_ascii=False, indent=2)
            for definition_list_wikt, definition_list_wiki in zip(definitions_wikt, definitions_wiki):
                definition_list = definition_list_wikt or definition_list_wiki
                filtered_definition_list = []
                for definition in definition_list:
                    definition, *examples = definition.split(self._examples_sep)
                    if self.filter_abbr:
                        definition = self.filter_intro_abbreviations(definition)
                    filtered_definition_list.append(definition)
                filtered_definitions.append(filtered_definition_list)
        return filtered_definitions

    def __call__(self, term_name: str) -> str:
        name = term_name.lower()
        name = re.sub('\s?\(.+?\)', '', name)
        defins = []
        if name in self.wikt_db and self.wikt_db[name] and any(len(d) for d in self.wikt_db[name]):
            defins.extend([d for d in self.wikt_db[name] if d.strip()])
        if name in self.wiki_db and self.wiki_db[name]:
            for d in self.wiki_db[name]:
                d = d.strip()
                if d:
                    if self.filter_wiki_brackets:
                        d = re.sub('\s?\(.+?\)', '', d)
                    d = self._get_first_sent(d)
                    defins.append(d)
        if not defins:
            defins.append(term_name)
        filtered_defins = []
        for defin in defins:
            defin, *examples = defin.split(self._examples_sep)
            defin = self.san.filter_diacritical(defin)
            defin = self.filter_intro_abbreviations(defin)
            filtered_defins.append(defin)
        return filtered_defins

    @staticmethod
    def _get_first_sent(text: str) -> str:
        return ru_sent_tokenize(text)[0]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', type=Path,
                        help='JSON file with dictionary, inference for'
                             ' keys of the dict',
                        default='/home/hdd/data/hypernym/train.cased.json')
    with open('/home/hdd/data/hypernym/synset_info.json', 'r', encoding='utf-8') as fin:
        synset_to_name = json.load(fin)

    ddb = DefinitionDB('/home/hdd/data/hypernym/wiki_wikt_db.json')
    words = []
    for fname in Path('/home/hdd/data/hypernym/').glob('*.tsv'):
        if 'candidates' in str(fname):
            continue
        with open(fname) as fin:
            for line in fin:
                word = line.lower().strip()
                # if word not in ddb.wikt_db and word not in ddb.wiki_db:
                words.append(word)
    ddb = DefinitionDB()
    for word in words[:10]:
        print(f'======= {word} ========')
        print(ddb(word))
    # #
    # print(words)
    # definitions = ddb.get_definition(words)
    # print(len(words))
    # for w in words:
    #     if len(ddb(w)) > 10:
    #         print(w)
    #         print(ddb(w))
    # print(Counter([len(ddb(w)) for w in words]).most_common())
