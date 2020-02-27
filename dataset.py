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

#                                   hyponym syns  hypernym syns
DATASET_TYPE = Dict[str, List[Tuple[List[str], List[List[str]]]]]


class HypoDataset(Dataset):
    def __init__(self,
                 tokenizer: BertTokenizer,
                 corpus_path: Union[str, Path],
                 hypo_index_path: Union[str, Path],
                 train_set_path: Union[str, Path],
                 hypernym_list: Union[List[Tuple[str]]],
                 debug: bool = False,
                 predict_all_hypes: bool = True,
                 max_len: int = 128,
                 valid_set_path: Union[str, Path, None] = None,
                 embed_with_special_tokens: bool = False,
                 level: str = 'sense',
                 sample_hypernyms: bool = False):
        self.tokenizer = tokenizer
        self.hypo_index = self._read_json(hypo_index_path)
        self.corpus = self._read_corpus(corpus_path, self.hypo_index)
        self.level = level
        self.embed_with_special_tokens
        self.sample_hypernyms = sample_hypernyms

        train_set = self._read_json(train_set_path)
        self.dataset = self._filter_dataset(train_set)
        self.train_set_idxs = np.arange(len(self.dataset))
        if valid_set_path is not None:
            valid_set = self._read_json(valid_set_path)
            valid_set = self._filter_dataset(valid_set)
            self.dataset.update(valid_set)
        self.valid_set_idxs = np.arange(len(self.train_set_idxs), len(self.dataset))
        # NOTE: here train keys() are not garanteed to return in the same order as added
        self.hypos = list(self.dataset)

        self.all_hypes_sense = {}
        self.all_hypes_synset = {}
        for hypo, hypos_hypes in self.dataset.items():
            hypo = hypo.lower()
            hypes_synset_level = []
            hypes_sense_level = []
            for _, hype_syns in hypos_hypes:
                hypes_synset_level.extend([tuple(hype_syn) for hype_syn in hype_syns])
                for hype_syn in hypes_synset_level:
                    hypes_sense_level.extend([(hype,) for hype in hype_syn])
            self.all_hypes_sense[hypo] = hypes_sense_level
            self.all_hypes_synset[hypo] = hypes_synset_level
        if sample_hypernyms:
            self.hypersynset2hypos = collections.defaultdict(list)
            for hypo, hyper_synsets in self.all_hypes_synset.items():
                for hyper_synset in hyper_synsets:
                    self.hypersynset2hypos[hyper_synset].append(hypo)
            self.hyper_synsets = list(self.hypersynset2hypos.keys())
            train_set_idxs = set(
                self.hyper_synsets.index(hyper_synset)
                for train_hypo_idx in self.train_set_idxs
                for hyper_synset in self.all_hypes_synset[self.hypos[train_hypo_idx]]
            )
            valid_set_idxs = set(range(len(self.hyper_synsets))) - train_set_idxs
            self.hypo_train_set_idxs = self.train_set_idxs
            self.hypo_valid_set_idxs = self.valid_set_idxs
            self.train_set_idxs = np.array(list(train_set_idxs))
            self.valid_set_idxs = np.array(list(valid_set_idxs))
        print(f'Train set contains {len(self.train_set_idxs)} samples.')
        print(f'Valid set contains {len(self.valid_set_idxs)} samples.')

        self.hypernym_to_idx = {hype: n for n, hype in enumerate(hypernym_list)}
        self.hypernym_list = hypernym_list
        self.debug = debug
        self.predict_all_hypes = predict_all_hypes
        self.max_len = max_len

    def get_train_idxs(self):
        return self.train_set_idxs

    def get_valid_idxs(self):
        return self.valid_set_idxs

    def get_valid(self):
        self.mode = 'valid'
        return self

    def _filter_dataset(self, dataset: DATASET_TYPE) -> DATASET_TYPE:
        filtered_dataset = {}
        not_in_index_count = 0
        for hypo, hypo_syns_and_hypes in dataset.items():
            hypo_syns_and_hypes: List[Tuple[List[str], List[List[str]]]]
            if self.hypo_index.get(hypo, []):
                filtered_dataset[hypo] = hypo_syns_and_hypes
            else:
                not_in_index_count += 1
        not_in_index_percent = not_in_index_count / (len(filtered_dataset) +
                                                     not_in_index_count)
        print(f'{not_in_index_percent:.2f} hyponyms are not found in the index')
        return filtered_dataset

    @classmethod
    def _read_json(cls, hypo_index_path: Union[str, Path]):
        with open(hypo_index_path, encoding='utf8') as handle:
            return json.load(handle)

    @staticmethod
    def _read_corpus(corpus_path: Union[str, Path],
                     index: Optional[Dict[str, List[Tuple[int]]]]=None) -> List[str]:
        print(f"Loading corpus from {corpus_path}.")
        with open(corpus_path, encoding='utf8') as handle:
            if index is not None:
                idxs = set(sent_idx for idxs in index.values() for sent_idx, _, _ in idxs)
                return [ln if i in idxs else '' for i, ln in tqdm.tqdm(enumerate(handle))]
            else:
                return handle.readlines()

    def __len__(self):
        if self.sample_hypernyms:
            return len(self.hyper_synsets)
        return len(self.dataset)

    def __getitem__(self, item):
        if self.sample_hypernyms:
            hypersynset = self.hyper_synsets[item]
            hypos = self.hypersynset2hypos[hypersynset]
            if item in self.train_set_idxs:
                hypos = [h for h in hypos
                         if self.hypos.index(h) in self.hypo_train_set_idxs]
            elif item in self.valid_set_idxs:
                hypos = [h for h in hypos
                         if self.hypos.index(h) in self.hypo_valid_set_idxs]
            else:
                raise RuntimeError(f"item {item} not in indeces.")
            hypo = choice(hypos).lower()
        else:
            hypo = self.hypos[item].lower()

        if self.level == 'sense':
            all_hypes = self.all_hypes_sense[hypo]
        elif self.level == 'synset':
            # TODO: mode tuplization to train
            all_hypes = self.all_hypes_synset[hypo]

        sent_idx, in_sent_start, in_sent_end = choice(self.hypo_index[hypo])
        sent_toks = self.corpus[sent_idx].split()
        subword_idxs, hypo_mask, subtok_start, subtok_end = \
            self._get_indices_and_masks(sent_toks, in_sent_start, in_sent_end)
        subword_idxs, hypo_mask, subtok_start, subtok_end = \
            self._cut_to_maximum_length(subword_idxs,
                                        hypo_mask,
                                        subtok_start,
                                        subtok_end,
                                        self.max_len)
        cls_idx, sep_idx = self.tokenizer.convert_tokens_to_ids(['[CLS]', '[SEP]'])
        subword_idxs = [cls_idx] + subword_idxs + [sep_idx]
        if self.embed_with_special_tokens:
            hypo_mask = [1.0] + hypo_mask + [1.0]
        else:
            hypo_mask = [0.0] + hypo_mask + [0.0]
        if sum(hypo_mask) == 0.0:
            print('Damn!')

        subword_idxs = subword_idxs[:self.max_len]
        hypo_mask = hypo_mask[:self.max_len]
        hype_prob = [0.0] * len(self.hypernym_list)
        if self.predict_all_hypes:
            hype_idxs = [self.hypernym_to_idx[tuple(hype)] for hype in all_hypes]
            single_hype_prob = 1 / len(hype_idxs)
            for hype_idx in hype_idxs:
                hype_prob[hype_idx] = single_hype_prob
        else:
            if self.sample_hypernyms:
                hype = hypersynset if self.level == 'synset' else choice(hypersynset)
            else:
                hype = choice(all_hypes)
            hype_idx = self.hypernym_to_idx[hype]
            hype_prob[hype_idx] = 1.0
        return subword_idxs, hypo_mask, hype_prob

    def _get_indices_and_masks(self,
                               sent_tokens: List[str],
                               in_sent_start: int,
                               in_sent_end: int) \
            -> Tuple[List[int], List[float], int, int]:
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

    @staticmethod
    def _cut_to_maximum_length(subword_idxs: List[str],
                               hypo_mask: List[str],
                               subtok_start: int,
                               subtok_end: int,
                               length: int) -> Tuple[List[str], List[str], int, int]:
        if len(subword_idxs) > length:
            half_len = length // 2
            new_start = max(0, subtok_start - half_len)
            subword_idxs = subword_idxs[new_start: new_start + length]
            hypo_mask = hypo_mask[new_start: new_start + length]

            new_subtok_start = subtok_start - new_start
            new_subtok_end = subtok_end - new_start
            return subword_idxs, hypo_mask, new_subtok_start, new_subtok_end
        return subword_idxs, hypo_mask, subtok_start, subtok_end

    def get_all_hypo_samples(self,
                             hypo: str) \
                             -> Tuple[List[Union[torch.Tensor, List[int], List[float]]]]:
        # TODO: is the method used anywhere?
        hypo_mentions = self.hypo_index[hypo]
        sents_indices, sents_hypo_masks = [], []
        for sent_idx, in_sent_start, in_sent_end  in hypo_mentions:
            sent_toks = self.corpus[sent_idx].split()
            subword_idxs, hypo_mask, subtok_start, subtok_end = \
                self._get_indices_and_masks(sent_toks, in_sent_start, in_sent_end)
            subword_idxs, hypo_mask, subtok_start, subtok_end = \
                        self._cut_to_maximum_length(subword_idxs,
                                                    hypo_mask,
                                                    subtok_start,
                                                    subtok_end,
                                                    self.max_len)
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


def get_indices_and_masks(sent_tokens: List[str],
                          in_sent_start: int,
                          in_sent_end: int,
                          tokenizer: BertTokenizer) \
        -> Tuple[List[int], List[float], int, int]:
    if in_sent_start not in range(len(sent_tokens)) or\
            in_sent_end not in range(1, len(sent_tokens) + 1):
        raise ValueError(f'wrong input: tokens {sent_tokens} don\'t contain pos'
                         f' ({in_sent_start}, {in_sent_end}).')
    sent_subword_idxs = []
    sent_subwords = []
    sent_hypo_mask = []
    new_in_sent_start, new_in_sent_end = None, None
    for n, tok in enumerate(sent_tokens):
        if n == in_sent_start:
            new_in_sent_start = len(sent_subwords)
        subtokens = tokenizer.tokenize(tok)
        sent_subwords.extend(subtokens)
        subtok_idxs = tokenizer.convert_tokens_to_ids(subtokens)
        sent_subword_idxs.extend(subtok_idxs)
        # NOTE: absence of + 1 because absence of [CLS] token in the beginning
        mask_value = float(in_sent_start <= n < in_sent_end)
        sent_hypo_mask.extend([mask_value] * len(subtok_idxs))
        if n == in_sent_end - 1:
            new_in_sent_end = len(sent_subwords) + 1
    return sent_subword_idxs, sent_hypo_mask, new_in_sent_start, new_in_sent_end


def get_hypernyms_list_from_train(train: Union[Path, str, List],
                                  level: str = 'sense') -> List[Tuple[str]]:
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
