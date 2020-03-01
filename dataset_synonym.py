#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import tqdm
import collections
from itertools import chain
from pathlib import Path
from random import choice
from typing import Union, List, Tuple, Optional, Iterable, Dict

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import BertTokenizer
import numpy as np

from dataset import get_indices_and_masks, HypoDataset

#             synset id, synonym_synset
DATASET_TYPE = Dict[str, List[str]]


class SynoDataset(Dataset):
    def __init__(self,
                 tokenizer: BertTokenizer,
                 corpus_path: Union[str, Path],
                 sense_index_path: Union[str, Path],
                 train_set_path: Union[str, Path],
                 output_synonym_list: List[Tuple[str]],
                 predict_one: bool = False,
                 embed_with_special_tokens: bool = True,
                 max_len: int = 128,
                 level: str = 'sense') -> None:
        self.tokenizer = tokenizer
        self.sense_index = self._read_json(sense_index_path)
        self.corpus = self._read_corpus(corpus_path, self.sense_index)
        self.output_synonym_list = output_synonym_list

        self.predict_one = predict_one
        self.embed_with_special_tokens = embed_with_special_tokens
        self.max_len = max_len
        self.level = level

        self.group2synsets = self._read_json(train_set_path)
        self.group2syno_senses = {gr: [phr for synset in syno_synsets for phr in synset]
                                  for gr, syno_synsets in self.group2synsets.items()}
        self.train_group2syno_senses = self._filter_dataset(self.group2syno_senses)
        self.train_groups = list(self.train_group2syno_senses.keys())

        num_train = round(len(self.train_groups) * 0.8)
        self.train_set_idxs = np.arange(num_train)
        self.valid_set_idxs = np.arange(num_train, len(self.train_groups))
        print(f'Train set contains {len(self.train_set_idxs)} samples.')
        print(f'Valid set contains {len(self.valid_set_idxs)} samples.')

    def __len__(self):
        return len(self.train_groups)

    def get_train_idxs(self):
        return self.train_set_idxs

    def get_valid_idxs(self):
        return self.valid_set_idxs

    def get_valid(self):
        self.mode = 'valid'
        return self

    def _filter_dataset(self, dataset: DATASET_TYPE) -> DATASET_TYPE:
        filtered_dataset = collections.defaultdict(list)
        for h_id, syno_senses in dataset.items():
            for phr in syno_senses:
                if self.sense_index.get(phr.lower(), []):
                    filtered_dataset[h_id].append(phr)
        not_in_index_percent = round(len(filtered_dataset) / len(dataset), 2) * 100
        print(f'{not_in_index_percent}% synonym groups are not found in the index.')
        return filtered_dataset

    @classmethod
    def _read_json(cls, sense_index_path: Union[str, Path]):
        with open(sense_index_path, encoding='utf8') as handle:
            return json.load(handle)

    @staticmethod
    def _read_corpus(corpus_path: Union[str, Path],
                     index: Optional[Dict[str, List[Tuple[int]]]] = None) -> List[str]:
        print(f"Loading corpus from {corpus_path}.")
        with open(corpus_path, encoding='utf8') as handle:
            if index is not None:
                idxs = set(sent_idx for idxs in index.values() for sent_idx, _, _ in idxs)
                return [ln if i in idxs else '' for i, ln in tqdm.tqdm(enumerate(handle))]
            else:
                return handle.readlines()

    def __getitem__(self, item):
        group = self.train_groups[item]
        input_synonym = choice(self.train_group2syno_senses[group])

        if self.level == 'sense':
            output_synonyms = [(sense,) for sense in self.group2syno_senses[group]]
        elif self.level == 'synset':
            output_synonyms = [tuple(synset) for synset in self.group2synsets[group]]

        sent_idx, in_sent_start, in_sent_end = \
            choice(self.sense_index[input_synonym.lower()])
        sent_toks = self.corpus[sent_idx].split()
        subword_idxs, syno_mask, subtok_start, subtok_end = \
            get_indices_and_masks(sent_toks, in_sent_start, in_sent_end, self.tokenizer)
        subword_idxs, syno_mask, subtok_start, subtok_end = \
            HypoDataset._cut_to_maximum_length(subword_idxs,
                                               syno_mask,
                                               subtok_start,
                                               subtok_end,
                                               self.max_len)
        cls_idx, sep_idx = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        subword_idxs = [cls_idx] + subword_idxs + [sep_idx]
        if self.embed_with_special_tokens:
            syno_mask = [1.0] + syno_mask + [1.0]
        else:
            syno_mask = [0.0] + syno_mask + [0.0]

        syno_idxs = [self.output_synonym_list.index(syno) for syno in output_synonyms]
        if self.predict_one:
            syno_idxs = [choice(syno_idxs)]
        output_probs = [1 / len(syno_idxs) if i in syno_idxs else 0.0
                        for i in range(len(self.output_synonym_list))]
        return subword_idxs, syno_mask, output_probs


def get_synonyms_list_from_train(train: Union[Path, str, List],
                                 level: str = 'sense') -> List[Tuple[str]]:
    if isinstance(train, (str, Path)):
        train = SynoDataset._read_json(train)
    synonym_set = set()
    for synonym_synsets in train.values():
        for synset in synonym_synsets:
            if level == 'sense':
                synonym_set.update((syno,) for syno in synset)
            elif level == 'synset':
                synonym_set.add(tuple(synset))
            else:
                raise NotImplementedError()
    return sorted(synonym_set)


def batch_collate(batch: List[Union[List[float], List[int], int]]) -> List[torch.Tensor]:
    """ Pads batch """
    indices, masks, syno_idxs = list(zip(*batch))
    indices, masks, attention_masks, syno_idxs = HypoDataset.torchify_and_pad(indices, masks, syno_idxs)
    return indices, masks, attention_masks, syno_idxs


if __name__ == '__main__':
    data_path = Path('/home/hdd/data/hypernym/')
    corpus_path = data_path / 'corpus.news_dataset-sample.token.txt'
    # TODO: build index for new train
    index_path = data_path / 'index.full.news_dataset-sample.json'
    train_set_path = data_path / 'train_synonyms.cased.not_lemma.json'

    model_path = Path('/home/hdd/models/rubert_cased_L-12_H-768_A-12_v2/')
    tokenizer_vocab_path = model_path / 'vocab.txt'

    tokenizer = BertTokenizer(tokenizer_vocab_path, do_lower_case=False)
    output_synonym_list = get_synonyms_list_from_train(train_set_path, level='sense')
    print(f'Synonym list: {output_synonym_list}')

    ds = SynoDataset(tokenizer,
                     corpus_path,
                     index_path,
                     train_set_path,
                     output_synonym_list,
                     predict_one=True,
                     level='sense')

    sentence_indices, sentence_syno_mask, syno_idx = next(iter(ds))
    print(f'Indices: {sentence_indices}\n'
          f'Input Synonym Mask: {sentence_syno_mask}\n'
          f'Synonym idx: {syno_idx}')
    print('=' * 20)
    print('Returning all synonyms')

    ds = SynoDataset(tokenizer,
                     corpus_path,
                     index_path,
                     train_set_path,
                     output_synonym_list,
                     predict_one=False,
                     level='sense')

    sentence_indices, sentence_syno_mask, syno_idxs = next(iter(ds))
    print(f'Indices: {sentence_indices}\n'
          f'Hypo Mask: {sentence_syno_mask}\n'
          f'syno_idxs: {syno_idxs}')
    print('=' * 20)

    dl = DataLoader(ds, batch_size=2, collate_fn=batch_collate)
    idxs_batch, mask_batch, attention_masks_batch, syno_idxs = next(iter(dl))
    print(f'Indices batch: {idxs_batch}\n'
          f'Mask batch: {mask_batch}\n'
          f'Attention mask batch: {attention_masks_batch}\n'
          f'Synonym indices batch: {syno_idxs}')

    print('='*20)
    synonym = 'кот'
    # batch = ds.get_all_syno_samples(synonym)
    # indices, masks, att_masks = batch
    # print(f'BERT inputs for all mentions of a synonym {synonym}\n'
    #       f'Indices: {indices}\n'
    #       f'Mask: {masks}\n'
    #       f'Attention masks: {att_masks}')
