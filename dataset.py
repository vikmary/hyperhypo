#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
from itertools import chain
from pathlib import Path
from random import choice
from typing import Union, List, Tuple, Optional, Iterable

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from transformers import BertTokenizer
import numpy as np

#                 hyponym syns  hyperonym syns  hyperhyperonym syns
DATASET_TYPE = List[Union[List[str], List[List[str]], List[List[str]]]]


class HypoDataset(Dataset):
    def __init__(self,
                 tokenizer: BertTokenizer,
                 corpus_path: Union[str, Path],
                 hypo_index_path: Union[str, Path],
                 train_set_path: Union[str, Path],
                 hypernym_list: List[str],
                 debug: bool = False,
                 predict_all_hypes=True,
                 max_len=128,
                 valid_set_path: Union[str, Path] = None,
                 level: str = 'sense'):
        self.tokenizer = tokenizer
        self.corpus = self._read_corpus(corpus_path)
        self.hypo_index = self._read_json(hypo_index_path)
        self.level = level

        train_set = self._read_json(train_set_path)
        self.dataset = self._filter_dataset(train_set)
        self.train_set_end_idx = len(self.dataset)
        if valid_set_path is not None:
            valid_set = self._read_json(valid_set_path)
            valid_set = self._filter_dataset(valid_set)
            self.dataset.update(valid_set)
        self.hypos = list(self.dataset)

        self.all_hypes_sense = {}
        self.all_hypes_synset = {}
        for hypo, hypos_hypes in self.dataset.items():
            hypo = hypo.lower()
            self.all_hypes_sense[hypo] = chain(*[hh[1] for hh in hypos_hypes])
            self.all_hypes_synset[hypo] = [tuple(hh[1]) for hh in hypos_hypes]

        self.hypernym_to_idx = {hype: n for n, hype in enumerate(hypernym_list)}
        self.hypernym_list = hypernym_list
        self.debug = debug
        self.predict_all_hypes = predict_all_hypes
        self.max_len = max_len

    def get_train_idxs(self):
        return np.arange(self.train_set_end_idx)

    def get_valid_idxs(self):
        return np.arange(self.train_set_end_idx, len(self.dataset))

    def get_valid(self):
        self.mode = 'valid'
        return self

    def _filter_dataset(self, dataset: DATASET_TYPE) -> DATASET_TYPE:
        filtered_dataset = []
        not_in_index_list = []
        for hypos, *hypes in dataset:
            for hypo in hypos:
                if hypo in self.hypo_index:
                    filtered_dataset.append([hypo, *hypes])
                else:
                    not_in_index_list.append(hypo)
        not_in_index_percent = len(not_in_index_list) / (
                    len(filtered_dataset) + len(not_in_index_list))
        print(f'{not_in_index_percent:.2f} hyponyms are not found in the index')
        return filtered_dataset

    @classmethod
    def _read_json(cls, hypo_index_path: Union[str, Path]):
        with open(hypo_index_path, encoding='utf8') as handle:
            return json.load(handle)

    @staticmethod
    def _read_corpus(corpus_path: Union[str, Path]):
        with open(corpus_path, encoding='utf8') as handle:
            return handle.readlines()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        senses_synsets = self.dataset[self.hypos[item]]
        hypo = self.hypos[item].lower()
        if self.level == 'sense':
            all_hypes = self.all_hypes_sense[hypo]
        elif self.level == 'synset':
            # TODO: mode tuplization to train
            all_hypes = self.all_hypes_synset[hypo]

        sent_idx, in_sent_start, in_sent_end = choice(self.hypo_index[hypo])
        sent_toks = self.corpus[sent_idx].split()
        sent_toks =  sent_toks
        subword_idxs, hypo_mask, subtok_start, subtok_end = \
            self._get_indices_and_masks(sent_toks, in_sent_start, in_sent_end)
        if len(subword_idxs) > self.max_len:
            half_max_len = self.max_len // 2
            new_start = max(0, subtok_start - half_max_len)
            subword_idxs = subword_idxs[new_start: new_start + self.max_len]
            hypo_mask = hypo_mask[new_start: new_start + self.max_len]
        cls_idx, sep_idx = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        subword_idxs = [cls_idx] + subword_idxs + [sep_idx]
        hypo_mask = [0.0] + hypo_mask + [0.0]
        if sum(hypo_mask) == 0.0:
            print('Damn!')

        subword_idxs = subword_idxs[:self.max_len]
        hypo_mask = hypo_mask[:self.max_len]
        hype_prob = [0.0] * len(self.hypernym_list)
        if self.predict_all_hypes:
            hype_idxs = [self.hypernym_to_idx[hype] for hype in all_hypes]
            single_hype_prob = 1 / len(hype_idxs)
            for hype_idx in hype_idxs:
                hype_prob[hype_idx] = single_hype_prob
        else:
            hype = choice(all_hypes)
            hype_idx = self.hypernym_to_idx[hype]
            hype_prob[hype_idx] = 1.0
        return subword_idxs, hypo_mask, hype_prob

    def _get_indices_and_masks(self,
                               sent_tokens: List[str],
                               in_sent_start: int,
                               in_sent_end: int):
        sent_subword_idxs = []
        sent_subwords = []
        sent_hypo_mask = []
        for n, tok in enumerate(sent_tokens):
            if n == in_sent_start:
                new_in_sent_start = len(sent_subwords)
            subtokens = self.tokenizer.tokenize(tok)
            sent_subwords.extend(subtokens)
            subtok_idxs = self.tokenizer.convert_tokens_to_ids(subtokens)
            sent_subword_idxs.extend(subtok_idxs)
            # NOTE: absence of + 1 because absence of [CLS] token in the beginning
            mask_value = float(in_sent_start <= n < in_sent_end)
            sent_hypo_mask.extend([mask_value] * len(subtok_idxs))
            if n == in_sent_end - 1:
                new_in_sent_end = len(sent_subwords) + 1
        return sent_subword_idxs, sent_hypo_mask, new_in_sent_start, new_in_sent_end

    def get_all_hypo_samples(self,
                             hypo: str) \
                             -> Tuple[List[Union[torch.Tensor, List[int], List[float]]]]:
        hypo_mentions = self.hypo_index[hypo]
        sents_indices, sents_hypo_masks = [], []
        for sent_idx, in_sent_start, in_sent_end  in hypo_mentions:
            sent_toks = self.corpus[sent_idx].split()
            subword_idxs, hypo_mask = self._get_indices_and_masks(sent_toks, in_sent_start, in_sent_end)
            sents_indices.append(subword_idxs)
            sents_hypo_masks.append(hypo_mask)
        batch_parts = self.torchify_and_pad(sents_indices, sents_hypo_masks)
        sents_indices, sents_hypo_masks, sents_att_masks = batch_parts
        return sents_indices, sents_hypo_masks, sents_att_masks

    @classmethod
    def torchify_and_pad(cls,
                         sents_indices: List[List[int]],
                         sents_masks: List[List[float]],
                         hype_one_hot: Optional[List[List[float]]] = None) -> Tuple[torch.Tensor]:
        batch_size = len(sents_indices)
        max_len = max(len(idx) for idx in sents_indices)
        padded_indices = torch.zeros(batch_size, max_len, dtype=torch.long)
        padded_mask = torch.zeros(batch_size, max_len, dtype=torch.float)
        padded_att_mask = torch.zeros(batch_size, max_len, dtype=torch.float)

        for n, (sent_idxs, sent_mask) in enumerate(zip(sents_indices, sents_masks)):
            up_to = len(sent_idxs)
            sent_idxs = torch.tensor(sent_idxs)
            sent_mask = torch.tensor(sent_mask)
            padded_indices[n, :up_to] = sent_idxs
            padded_mask[n, :up_to] = sent_mask
            padded_att_mask[n, :up_to] = 1.0
        if hype_one_hot:
            hype_one_hot = torch.tensor(hype_one_hot)
            return padded_indices, padded_mask, padded_att_mask, hype_one_hot
        else:
            return padded_indices, padded_mask, padded_att_mask


def get_hypernyms_list_from_train(train: Union[Path, str, List],
                                  level: str = 'sense') -> List[List[str]]:
    if isinstance(train, (str, Path)):
        train_set = HypoDataset._read_json(train)
    else:
        train_set = train
    hypernyms_set = set()
    for hypo, hypo_staff in train_set.items():
        for hypo_synset, hypes in hypo_staff:
            if level == 'sense':
                all_hypes = [(h, ) for h in chain(*hypes)]
            elif level == 'synset':
                all_hypes = [tuple(h) for h in hypes]
            else:
                raise NotImplementedError
            hypernyms_set.update(all_hypes)
    return sorted(hypernyms_set)


def batch_collate(batch: List[Union[List[float], List[int], int]]) -> List[torch.Tensor]:
    """ Pads batch """
    indices, masks, hype_idxs = list(zip(*batch))
    indices, masks, attention_masks, hype_idxs = HypoDataset.torchify_and_pad(indices, masks, hype_idxs)
    return indices, masks, attention_masks, hype_idxs


if __name__ == '__main__':
    # tokenizer_vocab_path = 'sample_data/vocab.txt'
    # corpus_path = 'sample_data/tst_corpus.txt'
    # hypo_index_path = 'sample_data/tst_index.json'
    # train_set_path = 'sample_data/tst_train.json'
    data_path = Path('/home/hdd/data/hypernym/')
    corpus_path = data_path / 'corpus.news_dataset-sample.token.txt'
    hypo_index_path = data_path / 'index.train.news_dataset-sample.json'
    train_set_path = data_path / 'train.cased.json'

    model_path = Path('/home/hdd/models/rubert_cased_L-12_H-768_A-12_v2/')
    tokenizer_vocab_path = model_path / 'vocab.txt'

    tokenizer = BertTokenizer(tokenizer_vocab_path, do_lower_case=False)
    hype_list = get_hypernyms_list_from_train(train_set_path)
    print(f'Hypernym list: {hype_list}')
    ds = HypoDataset(tokenizer,
                     corpus_path,
                     hypo_index_path,
                     train_set_path,
                     hype_list)

    sentence_indices, sentence_hypo_mask, hype_idx = next(iter(ds))
    print(f'Indices: {sentence_indices}\n'
          f'Hypo Mask: {sentence_hypo_mask}\n'
          f'Hype idx: {hype_idx}')
    print('=' * 20)
    print('Returning all hypes')

    ds = HypoDataset(tokenizer,
                     corpus_path,
                     hypo_index_path,
                     train_set_path,
                     hype_list,
                     predict_all_hypes=True)

    sentence_indices, sentence_hypo_mask, hype_idxs = next(iter(ds))
    print(f'Indices: {sentence_indices}\n'
          f'Hypo Mask: {sentence_hypo_mask}\n'
          f'hype_idxs: {hype_idxs}')
    print('=' * 20)

    dl = DataLoader(ds, batch_size=2, collate_fn=batch_collate)
    idxs_batch, mask_batch, attention_masks_batch, hype_idxs = next(iter(dl))
    print(f'Indices batch: {idxs_batch}\n'
          f'Mask batch: {mask_batch}\n'
          f'Attention mask batch: {attention_masks_batch}\n'
          f'Hype indices batch: {hype_idxs}')

    print('='*20)
    hyponym = 'кот'
    batch = ds.get_all_hypo_samples(hyponym)
    indices, masks, att_masks = batch
    print(f'BERT inputs for all mentions of a hyponym {hyponym}\n'
          f'Indices: {indices}\n'
          f'Mask: {masks}\n'
          f'Attention masks: {att_masks}')

# TODO: add repetition of the same hypo
# TODO: separate class
