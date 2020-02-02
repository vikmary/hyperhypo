#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import gzip
import fileinput
import zipfile
import functools
import unicodedata
from pathlib import Path
from contextlib import contextmanager
from typing import IO, Union, Optional, List, Iterator, Dict

import pymorphy2
from bs4 import BeautifulSoup


def count_lines(fpath: Union[str, Path]) -> int:

    def _blocks(files, size=65536):
        while True:
            b = files.read(size)
            if not b:
                break
            yield b

    with smart_open(fpath, "rt", encoding='utf-8', errors='ignore') as f:
        return sum(bl.count("\n") for bl in _blocks(f))


@contextmanager
def smart_open(p: Path, *args, **kwargs) -> IO:
    if '.gz' in p.suffixes:
        f = gzip.open(p, *args, **kwargs)
    else:
        f = open(p, *args, **kwargs)
    yield f
    f.close()


def extract_zip(p: Path) -> List[Path]:
    if '.zip' not in p.suffixes:
        raise ValueError(f"{p} is not a zip file.")
    with zipfile.ZipFile(p, 'r') as zip_obj:
        print(f"unzippping {p} into {p.parent}")
        zip_obj.extractall(p.parent)
        extracted_fpaths = [p.parent / n for n in zip_obj.namelist()
                            if (p.parent / n).is_file()]
    return extracted_fpaths


class Sanitizer:
    """Remove all combining characters like diacritical marks from utterance

    Args:
        diacritical: whether to remove diacritical signs or not
            diacritical signs are something like hats and stress marks
        nums: whether to replace all digits with 1 or not
    """

    def __init__(self,
                 filter_stresses: bool = True,
                 filter_empty_brackets: bool = True,
                 filter_diacritical: bool = False,
                 replace_nums_with: Optional[str] = None) -> None:
        self.do_filter_diacritical = filter_diacritical
        self.do_filter_empty_brackets = filter_empty_brackets
        self.do_filter_stresses = filter_stresses
        self.replace_nums_value = replace_nums_with
        self.nums_ptr = re.compile(r'[0-9]')
        self.stress_ptr = re.compile(r'́')
        self.whitespace_ptr = re.compile(r'\s+')
        self.brackets_ptr = re.compile(r'[\[\(]\s*[\]\)]')
        self.combining_characters = dict.fromkeys([c for c in range(sys.maxunicode)
                                                   if unicodedata.combining(chr(c))])

    def filter_duplicate_whitespaces(self, utterance: str) -> str:
        return self.whitespace_ptr.sub(' ', utterance)

    def filter_stresses(self, utterance: str) -> str:
        return self.stress_ptr.sub('', utterance)

    def filter_diacritical(self, utterance: str) -> str:
        return unicodedata.normalize('NFD', utterance)\
            .translate(self.combining_characters)

    def replace_nums(self, utterance: str, value: str = '1') -> str:
        return self.nums_ptr.sub(value, utterance)

    def filter_empty_brackets(self, utterance: str) -> str:
        return self.brackets_ptr.sub('', utterance)

    def __call__(self, utterance: str) -> str:
        if self.do_filter_stresses:
            utterance = self.filter_stresses(utterance)
        if self.do_filter_diacritical:
            utterance = self.filter_diacritical(utterance)
        if self.replace_nums_value is not None:
            utterance = self.replace_nums(utterance, self.replace_nums_value)
        if self.do_filter_empty_brackets:
            utterance = self.filter_empty_brackets(utterance)
        utterance = self.filter_duplicate_whitespaces(utterance)
        return utterance


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


class Lemmatizer:

    def __init__(self) -> None:
        self.lemmatizer = pymorphy2.MorphAnalyzer()

    @functools.lru_cache(maxsize=20000)
    def __call__(self, token: str) -> str:
        return self.lemmatizer.parse(token)[0].normal_form
