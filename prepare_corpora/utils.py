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
from typing import IO, Union, Optional, List, Iterator, Dict, Tuple

import pymorphy2
# import spacy
import spacy_udpipe


class Lemmatizer:

    def __init__(self, model: str) -> None:
        self.model = model
        if model == 'pymorphy':
            self.lemmatizer = pymorphy2.MorphAnalyzer()
        elif model == 'udpipe':
            try:
                udpipe_model = spacy_udpipe.load("ru")
                # udpipe_model = spacy_udpipe.UDPipeModel("ru")
            except Exception:
                print(f'Downloading udpipe ru model.')
                spacy_udpipe.download("ru")
                udpipe_model = spacy_udpipe.load("ru")
                # udpipe_model = spacy_udpipe.UDPipeModel("ru")
            self.lemmatizer = udpipe_model
            # self.lemmatizer = spacy.load("data/russian-syntagrus-ud-2.4-190531.udpipe",
            #                              udpipe_model=udpipe_model)
        else:
            raise ValueError('wrong model name for lemmatizer.')

    @functools.lru_cache(maxsize=20000)
    def __call__(self, tokens: List[str]) -> str:
        if model == 'pymorphy':
            return [self.lemmatizer.parse(token)[0].normal_form
                    for token in tokens]
        elif self.model == 'udpipe':
            return [token.lemma_ for token in self.lemmatizer(' '.join(tokens))]


class TextPreprocessor:
    def __init__(self,
                 model: str,
                 filter_stresses: bool = True,
                 filter_empty_brackets: bool = True,
                 lowercase: bool = True,
                 lemmatize: bool = True):
        self.model = model
        self.lowercase = lowercase
        self.lemmatize = lemmatize

        self.sanitizer = Sanitizer(filter_stresses=filter_stresses,
                                   filter_empty_brackets=filter_empty_brackets)
        if self.model == 'regexp+pymorphy':
            self.tokenizer = re.compile(r"[\w']+|[^\w ]")
            self.lemmatizer = Lemmatizer('pymorphy')
        elif self.model == 'udpipe':
            try:
                udpipe_model = spacy_udpipe.load('ru')
                # udpipe_model = spacy_udpipe.UDPipeModel("ru")
            except Exception:
                print(f'Downloading udpipe ru model.')
                spacy_udpipe.download("ru")
                udpipe_model = spacy_udpipe.load('ru')
                # udpipe_model = spacy_udpipe.UDPipeModel("ru")
            self.udpipe = udpipe_model
            # self.udpipe = spacy.load("data/russian-syntagrus-ud-2.4-190531.udpipe",
            #                          udpipe_model=udpipe_model)
        else:
            raise ValueError('wrong model name for preprocessor.')

    def __call__(self, text: str) -> Tuple[str, str]:
        text = self.sanitizer(text)

        if self.model == 'regexp+pymorphy':
            tokens = [token for token in self.tokenizer.findall(text)]
        elif self.model == 'udpipe':
            prep_text = self.udpipe(text)
            tokens = [token.text for token in prep_text]

        if self.lemmatize:
            if self.model == 'regexp+pymorphy':
                text = ' '.join(self.lemmatizer(token) for token in tokens)
            elif self.model == 'udpipe':
                text = ' '.join(token.lemma_ for token in prep_text)
        if self.lowercase:
            text = text.lower()
        return tokens, text


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
