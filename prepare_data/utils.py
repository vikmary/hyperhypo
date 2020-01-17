#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import sys
import gzip
import unicodedata
import functools
import pymorphy2
from pathlib import Path
from contextlib import contextmanager
from typing import IO, Union, Optional, List


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


class Sanitizer:
    """Remove all combining characters like diacritical marks from utterance

    Args:
        diacritical: whether to remove diacritical signs or not
            diacritical signs are something like hats and stress marks
        nums: whether to replace all digits with 1 or not
    """

    def __init__(self,
                 filter_diacritical: bool = True,
                 filter_empty_brackets: bool = False,
                 replace_nums_with: Optional[str] = None) -> None:
        self.do_filter_diacritical = filter_diacritical
        self.do_filter_empty_brackets = filter_empty_brackets
        self.replace_nums_value = replace_nums_with
        self.nums_ptr = re.compile(r'[0-9]')
        self.whitespace_ptr = re.compile(r'\s+')
        self.brackets_ptr = re.compile(r'[\[\(]\s*[\]\)]')
        self.combining_characters = dict.fromkeys([c for c in range(sys.maxunicode)
                                                   if unicodedata.combining(chr(c))])

    def filter_duplicate_whitespaces(self, utterance: str) -> str:
        return self.whitespace_ptr.sub(' ', utterance)

    def filter_diacritical(self, utterance: str) -> str:
        return unicodedata.normalize('NFD', utterance)\
            .translate(self.combining_characters)

    def replace_nums(self, utterance: str, value: str = '1') -> str:
        return self.nums_ptr.sub(value, utterance)

    def filter_empty_brackets(self, utterance: str) -> str:
        return self.brackets_ptr.sub('', utterance)

    def __call__(self, utterance: str) -> str:
        if self.do_filter_diacritical:
            utterance = self.filter_diacritical(utterance)
        if self.replace_nums_value is not None:
            utterance = self.replace_nums(utterance, self.replace_nums_value)
        if self.do_filter_empty_brackets:
            utterance = self.filter_empty_brackets(utterance)
        utterance = self.filter_duplicate_whitespaces(utterance)
        return utterance


class Lemmatizer:

    def __init__(self) -> None:
        self.lemmatizer = pymorphy2.MorphAnalyzer()

    @functools.lru_cache(maxsize=20000)
    def __call__(self, token: str) -> str:
        return self.lemmatizer.parse(token)[0].normal_form

