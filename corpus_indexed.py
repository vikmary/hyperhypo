#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import collections
import itertools
from tqdm import tqdm
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple, Set, Iterator

from utils import read_json_by_item
from prepare_corpora.utils import smart_open, count_lines


class CorpusIndexed:
    def __init__(self,
                 corpus_path: Union[str, Path],
                 index_path: Union[str, Path],
                 vocab: Optional[Iterator[str]] = None) -> None:
        self.corpus_path = Path(corpus_path)
        self.index_path = Path(index_path)

        self.idx, sent_idxs = self.load_index(self.index_path, vocab)
        self.corpus = self.load_corpus(self.corpus_path, sent_idxs)

    @classmethod
    def from_index(cls,
                   index_path: Union[str, Path],
                   vocab: Optional[Iterator[str]] = None) -> None:
        return cls(cls.get_corpus_path(index_path), index_path, vocab)

    @staticmethod
    def build_index(corpus_path: Union[str, Path],
                    vocab: Optional[Iterator[str]] = None,
                    max_utterances: Optional[int] = None,
                    max_utterances_per_item: Optional[int] = None)\
            -> Dict[str, List[str]]:
        # prioritize_with_max_length: int = 200
        max_utters_i = max_utterances_per_item
        if vocab is not None:
            is_frozen = True
            index = {phr: collections.Counter() for phr in vocab}
        else:
            is_frozen = False
            index = collections.defaultdict(collections.Counter)
        if max_utterances is None:
            print(f"Counting number of input lines in {corpus_path}.")
            max_utterances = count_lines(corpus_path)

        print(f"Building index for {corpus_path}.")
        with smart_open(corpus_path, 'rt') as utterances:
            for i_u, utter in tqdm(enumerate(utterances), total=max_utterances):
                if (i_u >= max_utterances) or \
                        (is_frozen and
                         all(len(v) >= (max_utters_i or 4) for v in index.values())):
                    break
                utter_lemmas = utter.split()
                for i_t, utter_lemma in enumerate(utter_lemmas):
                    if (not is_frozen or utter_lemma in index)\
                            and (not max_utters_i or
                                 (len(index[utter_lemma]) < max_utters_i)):
                        index[utter_lemma][(i_u, i_t, i_t+1)] += 1
                        # if len(utter_lemmas) < prioritize_with_max_length:
                        #     index[utter_lemma][(i_u, i_t)] += 1
                    if i_t + 1 < len(utter_lemmas):
                        utter_bilemma = ' '.join(utter_lemmas[i_t:i_t+2])
                        if (not is_frozen or utter_bilemma in index)\
                                and (not max_utters_i or
                                     (len(index[utter_bilemma]) < max_utters_i)):
                            index[utter_bilemma][(i_u, i_t, i_t+2)] += 1
                        if i_t + 2 < len(utter_lemmas):
                            utter_trilemma = ' '.join(utter_lemmas[i_t:i_t+3])
                            if (not is_frozen or utter_trilemma in index)\
                                    and (not max_utters_i or
                                         (len(index[utter_trilemma]) < max_utters_i)):
                                index[utter_trilemma][(i_u, i_t, i_t+3)] += 1
        index = {lemma: list(idxs) for lemma, idxs in index.items()}
        if vocab is not None:
            num_entries = len(list(itertools.chain(*index.values())))
            print(f"{len(index)} phrases, {num_entries} phrase contexts,"
                  f" {int(num_entries / len(index))} per phrase on average in index.")
            n_absent = sum(not len(v) for v in index.values())
            print(f"Haven't found context for {n_absent}/{len(index)}"
                  f" ({int(n_absent/len(index) * 100)}%) phrases from vocabulary.")
        return index

    @staticmethod
    def get_index_path(corpus_path: Union[str, Path],
                       suffix: str = 'full') -> Path:
        corpus_path = Path(corpus_path)
        base_name = corpus_path.name.rsplit('.txt', 1)[0]
        if 'corpus' in base_name:
            base_name = base_name.split('.', 1)[1]
        base_name = base_name.split('.', 1)[0]
        return corpus_path.with_name(f'index.{suffix}.{base_name}.json')

    @staticmethod
    def get_corpus_path(index_path: Union[str, Path],
                        level: str = 'token') -> Path:
        index_path = Path(index_path)
        base_name = index_path.name.rsplit('.json', 1)[0]
        base_name = base_name.split('.', 2)[-1]

        def corpus_path_with_fmt(fmt: '.txt.gz') -> Path:
            if level == 'token':
                return index_path.with_name(f'corpus.{base_name}.token{fmt}')
            elif level == 'lemma':
                return index_path.with_name(f'{base_name}.lemma{fmt}')

        corpus_path = corpus_path_with_fmt('.txt.gz')
        if not corpus_path.exists():
            corpus_path = corpus_path_with_fmt('.txt')
        if not corpus_path.exists() and ('-head-' in base_name):
            base_name = base_name.rsplit('-head-', 1)[0]
            corpus_path = corpus_path_with_fmt('.txt.gz')
            if not corpus_path.exists():
                corpus_path = corpus_path_with_fmt('.txt')
        if not corpus_path.exists():
            raise RuntimeError(f"Coudn't detect corpus path,"
                               f" corpus {corpus_path} doesn't exists.")
        return corpus_path

    @classmethod
    def load_corpus(cls,
                    path: Union[str, Path],
                    sent_idxs: Optional[Set[int]] = None) -> List[str]:
        print(f"Loading corpus from {path}.", file=sys.stderr)
        with smart_open(path, 'rt') as fin:
            if sent_idxs is not None:
                max_lines = max(sent_idxs)
                return [ln.strip() if i in sent_idxs else None
                        for i, ln in tqdm(zip(range(max_lines), fin), total=max_lines)]
            else:
                return [ln.strip() for ln in fin]

    @classmethod
    def load_index(cls,
                   path: Union[str, Path],
                   vocab: Optional[Iterator[str]] = None) \
            -> Tuple[Dict[str, collections.Counter], Set[int]]:
        print(f"Loading index from {path}.", file=sys.stderr)

        index = {}
        sent_idxs = set()
        if vocab is not None:
            for lemma in vocab:
                index[lemma] = collections.Counter()
            print(f"Will load only {len(index)} lemmas present in a vocab.")
            for lemma, idxs in read_json_by_item(open(path, 'rt')):
                if lemma in index:
                    for idx in idxs:
                        index[lemma][tuple(idx)] += 1
                        sent_idxs.add(idx[0])
        else:
            for lemma, idxs in read_json_by_item(open(path, 'rt')):
                if lemma not in index:
                    index[lemma] = collections.Counter()
                for idx in idxs:
                    index[lemma][tuple(idx)] += 1
                    sent_idxs.add(idx[0])
        return index, sent_idxs

    def get_contexts(self,
                     lemma: str,
                     max_num_tokens: Optional[int] = None) -> List[Tuple[str, int]]:
        if lemma not in self.idx:
            print(f"Warning: lemma '{lemma}' not in index.", file=sys.stderr)
            return []
        sents = ((self.corpus[sent_idx].split(), l_start, l_end)
                 for sent_idx, l_start, l_end in self.idx[lemma])
        if max_num_tokens is not None:
            sents = list(filter(lambda s_pos: len(s_pos[0]) < max_num_tokens, sents))
            if not sents:
                w_size = max_num_tokens // 2
                sents = ((self.corpus[s_id].split()[l_start - w_size: l_start + w_size],
                          w_size - max(w_size - l_start, 0),
                          w_size - max(w_size - l_start, 0) + (l_end - l_start))
                         for s_id, l_start, l_end in self.idx[lemma])
        return list(sents)
