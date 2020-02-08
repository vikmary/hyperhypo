#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import collections
from pathlib import Path
from typing import Optional, Union, List, Dict, Tuple

from utils import read_json_by_item
from prepare_corpora.utils import smart_open


class CorpusIndexed:
    def __init__(self,
                 index_path: Union[str, Path],
                 vocab: Optional[List[str]] = None) -> None:
        self.vocab = vocab
        self.index_path = Path(index_path)

        base_name = self.index_path.name.rsplit('.json', 1)[0]
        base_name = base_name.split('.', 2)[-1]
        if '-head-' in base_name:
            base_name = base_name.rsplit('-head-', 1)[0]
        self.corpus_path = Path(self.index_path.with_name('corpus.' + base_name +
                                                          '.token.txt.gz'))
        if not self.corpus_path.exists():
            self.corpus_path = Path(self.index_path.with_name('corpus.' + base_name +
                                                              '.token.txt'))
        if not self.corpus_path.exists():
            raise RuntimeError(f"corpus {self.corpus_path} doesn't exists")

        self.idx, sent_idxs = self.load_index(self.index_path, self.vocab)
        self.corpus = self.load_corpus(self.corpus_path, sent_idxs)

    @classmethod
    def load_corpus(cls,
                    path: Union[str, Path],
                    sent_idxs: Optional[set] = None) -> List[str]:
        print(f"Loading corpus from {path}.", file=sys.stderr)
        with smart_open(path, 'rt') as fin:
            if sent_idxs is not None:
                return [ln.strip() if i in sent_idxs else None
                        for i, ln in enumerate(fin)]
            else:
                return [ln.strip() for ln in fin]

    @classmethod
    def load_index(cls,
                   path: Union[str, Path],
                   vocab: Optional[List[str]] = None) -> Dict[str, collections.Counter]:
        print(f"Loading index from {path}.", file=sys.stderr)

        index = {}
        sent_idxs = set()
        if vocab is not None:
            print(f"Will load only {len(vocab)} lemmas present in a vocab.")
            for lemma in vocab:
                index[lemma] = collections.Counter()
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
        sents = ((self.corpus[sent_idx].split(), pos)
                 for sent_idx, pos in self.idx[lemma])
        if max_num_tokens is not None:
            sents = list(filter(lambda s_pos: len(s_pos[0]) < max_num_tokens, sents))
            if not sents:
                w_size = max_num_tokens // 2
                sents = ((self.corpus[s_id].split()[pos - w_size: pos + w_size],
                          w_size - max(w_size - pos, 0))
                         for s_id, pos in self.idx[lemma])
        return list(sents)
