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

from dataset import get_indices_and_mask

#             synset id, synonym_synset
DATASET_TYPE = Dict[str, List[str]]


class SynoDataset(Dataset):
    def __init__(self,
                 tokenizer: BertTokenizer,
                 corpus_path: Union[str, Path],
                 sense_index_path: Union[str, Path],
                 train_set_path: Union[str, Path],
                 output_synonym_list: List[Tuple[str]],
                 embed_with_special_tokens: bool = False,
                 max_len: int = 128,
                 level: str = 'sense') -> None:
        self.tokenizer = tokenizer
        self.sense_index = self._read_json(sense_index_path)
        self.corpus = self._read_corpus(corpus_path, self.sense_index)
        self.output_synonym_list = output_synonym_list

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
                if self.sense_index.get(phr, []):
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
                     index: Optional[Dict[str, List[Tuple[int]]]]=None) -> List[str]:
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
            output_synonyms = self.group2syno_senses[group]
        elif self.level == 'synset':
            output_synonyms = self.group2syno_synsets[group]

        sent_idx, in_sent_start, in_sent_end = choice(self.sense_index[input_synonym])
        sent_toks = self.corpus[sent_idx].split()
        subword_idxs, syno_mask, subtok_start, subtok_end = \
            get_indices_and_masks(sent_toks, in_sent_start, in_sent_end)
        subword_idxs, syno_mask, subtok_start, subtok_end = \
            self._cut_to_maximum_length(subword_idxs,
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

        hype_prob = [0.0] * len(self.hypernym_list)
        syno_probs = [0.0] * len(self.output_synonym_list)
        if self.predict_all_hypes:
            syno_idxs = [self.output_synonym_list.index(syno) for syno in output_synonyms]
        else:
            random_output_syno = choice(output_synonyms)
            syno_idxs = [self.output_synonym_list.index(random_output_syno)]
        single_syno_prob = 1 / len(syno_idxs)
        for syno_idx in syno_idxs:
            syno_probs[syno_idx] = single_syno_prob
        return subword_idxs, syno_mask, hype_prob

    @staticmethod
    def _cut_to_maximum_length(subword_idxs: List[str],
                               syno_mask: List[str],
                               subtok_start: int,
                               subtok_end: int,
                               length: int) -> Tuple[List[str], List[str], int, int]:
        if len(subword_idxs) > length:
            half_len = length // 2
            new_start = max(0, subtok_start - half_len)
            subword_idxs = subword_idxs[new_start: new_start + length]
            syno_mask = syno_mask[new_start: new_start + length]

            new_subtok_start = subtok_start - new_start
            new_subtok_end = subtok_end - new_start
            return subword_idxs, syno_mask, new_subtok_start, new_subtok_end
        return subword_idxs, syno_mask, subtok_start, subtok_end

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
                                  level: str = 'sense') -> List[Tuple[str]]:
    if isinstance(train, (str, Path)):
        train_set = HypoDataset._read_json(train)
    else:
        train_set = train
    hypernyms_set = set()
    for syno, syno_staff in train_set.items():
        for syno_synset, hypes in syno_staff:
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
    # sense_index_path = 'sample_data/tst_index.json'
    # train_set_path = 'sample_data/tst_train.json'
    data_path = Path('/home/hdd/data/hypernym/')
    corpus_path = data_path / 'corpus.news_dataset-sample.token.txt'
    sense_index_path = data_path / 'index.train.news_dataset-sample.json'
    train_set_path = data_path / 'train.cased.json'

    model_path = Path('/home/hdd/models/rubert_cased_L-12_H-768_A-12_v2/')
    tokenizer_vocab_path = model_path / 'vocab.txt'

    tokenizer = BertTokenizer(tokenizer_vocab_path, do_lower_case=False)
    hype_list = get_hypernyms_list_from_train(train_set_path)
    print(f'Hypernym list: {hype_list}')
    ds = HypoDataset(tokenizer,
                     corpus_path,
                     sense_index_path,
                     train_set_path,
                     hype_list)

    sentence_indices, sentence_syno_mask, hype_idx = next(iter(ds))
    print(f'Indices: {sentence_indices}\n'
          f'Hypo Mask: {sentence_syno_mask}\n'
          f'Hype idx: {hype_idx}')
    print('=' * 20)
    print('Returning all hypes')

    ds = HypoDataset(tokenizer,
                     corpus_path,
                     sense_index_path,
                     train_set_path,
                     hype_list,
                     predict_all_hypes=True)

    sentence_indices, sentence_syno_mask, hype_idxs = next(iter(ds))
    print(f'Indices: {sentence_indices}\n'
          f'Hypo Mask: {sentence_syno_mask}\n'
          f'hype_idxs: {hype_idxs}')
    print('=' * 20)

    dl = DataLoader(ds, batch_size=2, collate_fn=batch_collate)
    idxs_batch, mask_batch, attention_masks_batch, hype_idxs = next(iter(dl))
    print(f'Indices batch: {idxs_batch}\n'
          f'Mask batch: {mask_batch}\n'
          f'Attention mask batch: {attention_masks_batch}\n'
          f'Hype indices batch: {hype_idxs}')

    print('='*20)
    synonym = 'кот'
    batch = ds.get_all_syno_samples(synonym)
    indices, masks, att_masks = batch
    print(f'BERT inputs for all mentions of a synonym {synonym}\n'
          f'Indices: {indices}\n'
          f'Mask: {masks}\n'
          f'Attention masks: {att_masks}')

# TODO: add repetition of the same syno
# TODO: separate class
